//
// Created by alan on 06/04/23.
//
#pragma once
#ifndef CUSTOMNN_NUERAL_NETWORK_H
#define CUSTOMNN_NUERAL_NETWORK_H

#include <vector>
#include "matrix.h"

class NueralNetwork{
public:
    std::vector<Matrix> weightMatrixs;
    std::vector<Matrix> gradientMatrixs;
    std::vector<Matrix> outputMatrixs;

    std::vector<int> structure;

    NueralNetwork(int numInput,const std::vector<int>& hiddenLayerStructure,int numOutput);

    Matrix feedFoward(const Matrix &in);

    void randomizeWeightMatrix();

    void updateWeights(Matrix &in, Matrix &actual);

    void backprop(const Matrix &cost);
};

Matrix mseLossDerivitive(const Matrix &pred,const Matrix &actual);
#endif //CUSTOMNN_NUERAL_NETWORK_H
