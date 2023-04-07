//
// Created by alan on 06/04/23.
//
#include "../include/nueral_network.h"
#include <cmath>

NueralNetwork::NueralNetwork(int numInput, const std::vector<int>& hiddenLayerStructure,int numOutput) {
    this->structure = std::vector<int>();
    structure.push_back(numInput);

    for(const int &i : hiddenLayerStructure)
        structure.push_back(i);

    structure.push_back(numOutput);

    weightMatrixs = std::vector<Matrix>();


    int previous = structure.at(0);
    for(int i = 1; i < structure.size(); i++){
        std::cout << "adding\n";
        weightMatrixs.emplace_back(structure.at(i),previous);
        previous = structure.at(i);
    }
}

void NueralNetwork::randomizeWeightMatrix() {
    for (Matrix &weightMatrix: weightMatrixs) {
        randomizeMatrix(weightMatrix);
    }
}

// todo: make this faster by removing calls to the heap.
Matrix NueralNetwork::feedFoward(const Matrix &in) {
    /**
     * how does this work?
     * takes in input multiplies by weight matrix, which gives output pass that through layers till you've finished
     *
     * output matrix (weightMatrix.rows,1)
     * matrixMult(weightMatrix,current,output)
     * */
     Matrix tempOutput = multiplyMatrix(weightMatrixs.at(0),in);
     for(int layer = 1; layer < weightMatrixs.size(); layer++){
         const Matrix &weightMatrix = weightMatrixs.at(layer);
         tempOutput = multiplyMatrix(weightMatrix,tempOutput);
     }

    return  tempOutput;
}

void NueralNetwork::updateWeights(Matrix &in, Matrix &actual) {
    double h = 0.001;
    for(Matrix& weightMatrix: weightMatrixs){
        for(int i = 0; i < weightMatrix.rows * weightMatrix.columns;i++){
            Matrix initialCost =  mseLoss(feedFoward(in),actual);
            weightMatrix.setRawElement(i,weightMatrix.getRawElement(i) + h);

            Matrix afterNudge = mseLoss(feedFoward(in),actual);
            weightMatrix.setRawElement(i,weightMatrix.getRawElement(i) - h);
            Matrix derivitive = minusMatrix(afterNudge,initialCost);

            weightMatrix.setRawElement(i,(weightMatrix.getRawElement(i) - (derivitive.getRawElement(0) / h) * 0.1));
        }
    }

}

Matrix mseLoss(const Matrix &pred, const Matrix &actual){
    if(pred.rows != actual.rows || pred.columns != actual.columns){
        std::cout << "ERROR non equal matrix size for mseLoss calculation\n";
        exit(1);
    }

    Matrix lossMatrix = Matrix(pred.rows,pred.columns);
    for(int i = 0; i < pred.rows * pred.columns; i++){
        lossMatrix.setRawElement(i,pow(pred.getRawElement(i) - actual.getRawElement(i),2));
    }

    return lossMatrix;
}

