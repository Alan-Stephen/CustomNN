#include <iostream>
#include "matrix.h"
#include <vector>
#include "nueralnetwork.h"
#include <chrono>


void MatrixMultTest(int shared,int a,int b);

int main(int argc, char const *argv[])
{
	auto start = std::chrono::high_resolution_clock::now();
	
//	int structure[5] = {3,5,3,4,3};
//	NueralNetwork network(structure,5);
	for(int i = 0;i < 500;i++){
		MatrixMultTest(rand() % 100,rand() % 100,rand() % 100);
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto actualstart =  std::chrono::time_point_cast<std::chrono::microseconds>(start).time_since_epoch().count();
	auto actualend =  std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count();
	std::cout<< "BENCHMARKS" << (actualend - actualstart) * 0.001;
} 

void MatrixMultTest(int shared,int a,int b) {
	RandomMatrix matrix(a,shared);
	RandomMatrix matrix2(shared,b);
	Matrix output = matrix * matrix2;
	return;

}
