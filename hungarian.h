/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * Copyright (c) 2017, Reagents of the University of California                    *
 * Author: William C. Lenthe                                                       *
 * All rights reserved.                                                            *
 *                                                                                 *
 * Redistribution and use in source and binary forms, with or without              *
 * modification, are permitted provided that the following conditions are met:     *
 *                                                                                 *
 * 1. Redistributions of source code must retain the above copyright notice, this  *
 *    list of conditions and the following disclaimer.                             *
 *                                                                                 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,    *
 *    this list of conditions and the following disclaimer in the documentation    *
 *    and/or other materials provided with the distribution.                       *
 *                                                                                 *
 * 3. Neither the name of the copyright holder nor the names of its                *
 *    contributors may be used to endorse or promote products derived from         *
 *    this software without specific prior written permission.                     *
 *                                                                                 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"     *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE       *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  *
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE    *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL      *
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR      *
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      *
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   *
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE   *
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.            *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef _hungarian_h_
#define _hungarian_h_

#include <vector>
#include <numeric>//iota
#include <limits>
#include <algorithm>
#include <functional>//bind
#include <thread>

template <typename T>
class Hungarian {
	public:
		Hungarian(const size_t rows, const size_t columns) :
			numCol(std::max(rows, columns)),//convert up to square matrix
			numRow(std::max(rows, columns)),//convert up to square matrix
			vFill(std::numeric_limits<T>::infinity()),//get value to fill extra rows/columns from conversion to square matrix
			colCovered(numRow, false),//flag for if each row / column is uncovered
			rowCovered(numRow, false),//flag for if each row / column is uncovered
			cost(numRow * numCol, vFill),//cost of each assignment in row major order
			index(numRow),//indexing into cost matrix
			zeros(numRow),//set of columns that are currently 0 in each row
			primedZeros(numRow),//column of the currently primed zero in each row (or -1 for no starred zero)
			starredZeros(numRow, static_cast<size_t>(-1)),//column of the currently starred zero in each row (or -1 for no starred zero)
			workers(std::max<unsigned>(std::thread::hardware_concurrency(), 1)),//thread pool for parallel operations
			workerRows(workers.size() + 1, 0),//start/stop rows for each thread in pool
			workerCosts(workers.size())//working output for each thread in pool
		{
			static_assert(std::numeric_limits<T>::has_infinity, "infinity required for cost matrix initialization");
			std::iota(index.begin(), index.end(), 0);
			std::transform(index.begin(), index.end(), index.begin(), std::bind(std::multiplies<size_t>(), std::placeholders::_1, numCol));
		}

		void setCost(const size_t row, const size_t col, T c) {cost[index[row] + col] = c;}

		std::vector<size_t> compute() {
			//subtract minimum cost from each row and find minimum cost in each column
			std::vector<T> colMin(numCol, vFill);
			for(size_t row = 0; row < numRow; row++) {
				T h = *std::min_element(cost.begin() + index[row], cost.begin() + index[row] + numCol);
				if(h != vFill)
					std::transform(cost.begin() + index[row], cost.begin() + index[row] + numCol, cost.begin() + index[row], std::bind(std::minus<T>(), std::placeholders::_1, h));
				std::transform(cost.begin() + index[row], cost.begin() + index[row] + numCol, colMin.begin(), colMin.begin(), [](const T a, const T b){return std::min(a, b);});
			}

			//subtract minimum cost from each column
			std::transform(colMin.begin(), colMin.end(), colMin.begin(), [this](const T v){return v == vFill ? static_cast<T>(0) : v;});
			for(size_t row = 0; row < numRow; row++)
				std::transform(cost.begin() + index[row], cost.begin() + index[row] + numCol, colMin.begin(), cost.begin() + index[row], std::minus<T>());
			
			//star each zero if there is no starred zero in its row or column
			for(size_t row = 0; row < numRow; row++) {
				for(size_t col = 0; col < numCol; col++) {
					if(static_cast<T>(0) == cost[index[row] + col]) {
						zeros[row].push_back(col);
						if(!rowCovered[row] && !colCovered[col]) {
							rowCovered[row] = true;
							colCovered[col] = true;
							starredZeros[row] = col;
						}
					}
				}
			}

			//cover each column containing a stared zero
			std::fill(rowCovered.begin(), rowCovered.end(), false);
			std::fill(colCovered.begin(), colCovered.end(), false);
			for(const size_t c : starredZeros) {
				if(static_cast<size_t>(-1) != c)
					colCovered[c] = true;
			}

			//compute assignments
			while(!std::all_of(colCovered.begin(), colCovered.end(), [](const bool b){return b;}))
				iterate();
			return starredZeros;
		}

	private:
		void rebalanceWorkers() {
			//the covered rows change regularly and extra threads are wasted if uncovered rows aren't evenly distributed
			const size_t uncoveredRows = std::count(rowCovered.begin(), rowCovered.end(), false);//get number of uncovered rows to be distributed
			const size_t rowsPerWorker = (size_t) std::ceil(uncoveredRows / workers.size());//round up to get number of rows per worker thread
			size_t ind = 1;//workerRows[0] is always 0
			size_t rowCount = 0;
			for(size_t row = 0; row < numRow; row++) {
				if(!rowCovered[row]) {
					if(rowsPerWorker == rowCount++) {
						workerRows[ind++] = row;
						rowCount = 0;
					}
				}
			}
			for(size_t i = ind; i < workerRows.size(); i++) workerRows[i] = numRow;//for small numbers of rows there may be unused threads
		}

		void fillColInds() {
			//since the columns are iterated over so many times there is an appreciable performance gain from maintaining a list of uncovered indices
			colInds.clear();
			for(size_t col = 0; col < numCol; col++) {
				if(!colCovered[col])
					colInds.push_back(col);
			}
		}

		bool findUncoveredZero(size_t& row, size_t& col) const {
			//this operation acounts for ~10% of execution time
			for(row = 0; row < numRow; row++) {
				if(!rowCovered[row]) {
					for(const size_t& c : zeros[row]) {
						if(!colCovered[c]) {
							col = c;
							return true;
						}
					}
				}
			}
			return false;
		}

		T findMinUncoveredCost() {
			//this operation acounts for ~25% of execution time so it is worth running in parallel
			std::fill(workerCosts.begin(), workerCosts.end(), vFill);
			for(size_t i = 0; i < workers.size(); i++) workers[i] = std::thread(&Hungarian::findMinUncoveredCostWorker, this, i);
			for(size_t i = 0; i < workers.size(); i++) workers[i].join();
			return *std::min_element(workerCosts.begin(), workerCosts.end());
		}

		void findMinUncoveredCostWorker(const size_t i) {
			for(size_t row = workerRows[i]; row < workerRows[i+1]; row++) {
				if(!rowCovered[row]) {
					for(const size_t& col : colInds)
						if(cost[index[row] + col] < workerCosts[i]) workerCosts[i] = cost[index[row] + col];
				}
			}
		}

		void updateCosts(const T h) {
			//this operation acounts for ~60% of execution time so it is worth running in parallel
			for(size_t i = 0; i < workers.size(); i++) workers[i] = std::thread(&Hungarian::updateCostsWorker, this, h, i);
			for(size_t i = 0; i < workers.size(); i++) workers[i].join();
		}

		void updateCostsWorker(const T h, const size_t i) {
			for(size_t row = workerRows[i]; row < workerRows[i+1]; row++) {
				if(rowCovered[row]) {
					//add minimum uncovered value to each covered row
					std::transform(cost.begin() + index[row], cost.begin() + index[row] + numCol, cost.begin() + index[row], std::bind(std::plus<T>(), std::placeholders::_1, h));
					zeros[row].clear();
				}
				for(const size_t& col : colInds) {
					//subtract minimum uncovered value from each uncovered column
					if(static_cast<T>(0) == (cost[index[row] + col] -= h))
						zeros[row].push_back(col);
				}
			}

		}

		bool findStarredZeroInCol(size_t& row, const size_t col) const {
			for(row = 0; row < numRow; row++) {
				if(starredZeros[row] == col)
					return true;
			}
			return false;
		}

		void iterate() {
			size_t r, c;
			while(findUncoveredZero(r, c)) {
				primedZeros[r] = c;
				if(starredZeros[r] == static_cast<size_t>(-1)) {
					//build a sequence of alternating primed/starred zeros starting with primed zero if row doesn't contain a star
					rowSequence.resize(1);
					colSequence.resize(1);
					rowSequence[0] = r;
					colSequence[0] = c;

					//while a stared zero exists in the colum of the current primed zero
					while(findStarredZeroInCol(r, c)) {
						c = primedZeros[r];
						rowSequence.push_back(r);
						colSequence.push_back(c);
					}

					//unstar each stared zero in star sequence and star each primed zero in sequence
					for(size_t i = 0; i < colSequence.size(); i++)
						starredZeros[rowSequence[i]] = colSequence[i];

					//uncover all rows/columns and erase all primes (don't need to change primedZeros since they'll be overwritten)
					std::fill(rowCovered.begin(), rowCovered.end(), false);
					std::fill(colCovered.begin(), colCovered.end(), false);

					//cover all columns with starred zeros
					for(size_t r = 0; r < numRow; r++) {
						if(starredZeros[r] != static_cast<size_t>(-1))
							colCovered[starredZeros[r]] = 1;
					}
					return;
				}

				//cover row/uncover column of starred zero
				rowCovered[r] = true;
				colCovered[starredZeros[r]] = false;

			}
			//get indicies of uncovered columns and rebalance load on worker threads
			fillColInds();
			rebalanceWorkers();

			//if there are no uncovered zeros find minimum uncovered value and update cost matrix
			const T h = findMinUncoveredCost();
			if(h == vFill) {//if minimum is fill value we've run out of valid assignments
				std::fill(colCovered.begin(), colCovered.end(), true);
				return;
			}
			updateCosts(h);//update cost matrix
		}
		
		const size_t numCol, numRow;
		const T vFill;
		std::vector<bool> colCovered, rowCovered;
		std::vector<T> cost;
		std::vector<size_t> index;
		std::vector< std::vector<size_t> > zeros;
		std::vector<size_t> primedZeros, starredZeros;
		std::vector<size_t> rowSequence, colSequence;
		std::vector<size_t> colInds;

		std::vector<std::thread> workers;
		std::vector<size_t> workerRows;
		std::vector<T> workerCosts;
};

#endif