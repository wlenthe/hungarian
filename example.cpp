#include <iostream>
#include <random>

#include "hungarian.h"

int main() {
	//seed random number generator
	std::random_device device;
	std::mt19937 engine(device());
	std::uniform_real_distribution<float> dist(0, 100);

	//create cost solver and cost matrix
	size_t numSinks = 6, numSources = 6;
	Hungarian<float> h(numSinks, numSources);
	for(int j = 0; j < numSources; j++)
		std::cout << "\t   " << j;
	std::cout << "\n";
	for(int i = 0; i < numSinks; i++) {
		std::cout << "  " << i << "\t";
		for(int j = 0; j < numSources; j++) {
			float v = dist(engine);
			h.setCost(i, j, v);
			std::cout << v << "\t";
		}
		std::cout << "\n";
	}

	//solve and print assignments
	std::vector<size_t> matches = h.compute();
	std::cout << "\nsink/source:\n";
	for(size_t i = 0; i < numSinks; i++) {
		std::cout << i << " <- ";
		if(matches[i] == static_cast<size_t>(-1))
		 std::cout << "x\n";
		else
			std::cout << matches[i] << "\n";
	}
	return 0;
}