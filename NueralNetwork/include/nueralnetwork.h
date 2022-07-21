#pragma once
#include "matrix.h"
#include "vector"
#include <math.h>

//TODO CREATE NEW SUBCLASS OF MATRIX WHICH HAS RANDOM START VALUES?
class NueralNetwork {
    
    std::vector<RandomMatrix> weightMatrix;
    std::vector<RandomMatrix> biasMatrix;
    std::vector<int> structure;
    
    double activate(double x);

    public:
    Matrix feedFoward(Matrix a);
    NueralNetwork(const int* structure, int depth);   
};