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
    std::vector<int> structure;

    NueralNetwork(int numInput,const std::vector<int>& hiddenLayerStructure,int numOutput);

    Matrix feedFoward(const Matrix &in);

    void randomizeWeightMatrix();

    void updateWeights(Matrix &in, Matrix &actual);
};

Matrix mseLoss(const Matrix &pred, const Matrix &actual);
#endif //CUSTOMNN_NUERAL_NETWORK_H
