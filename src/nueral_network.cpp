//
// Created by alan on 06/04/23.
//
#include "../include/nueral_network.h"
#include <cmath>

// todo: remove input and output weights?
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
        gradientMatrixs.emplace_back(structure.at(i),previous);
        previous = structure.at(i);
    }
}

void NueralNetwork::randomizeWeightMatrix() {
    for (Matrix &weightMatrix: weightMatrixs) {
        randomizeMatrix(weightMatrix);
    }
}

Matrix NueralNetwork::feedFoward(const Matrix &in) {
    /**
     * how does this work?
     * takes in input multiplies by weight matrix, which gives output pass that through layers till you've finished
     *
     * output matrix (weightMatrix.numRows,1)
     * matrixMult(weightMatrix,current,output)
     * */
     Matrix tempOutput = multiplyMatrix(weightMatrixs.at(0),in);
    copyMatrix(tempOutput,outputMatrixs.at(0));
     for(int layer = 1; layer < weightMatrixs.size(); layer++){
         const Matrix &weightMatrix = weightMatrixs.at(layer);
         tempOutput = multiplyMatrix(weightMatrix,tempOutput);
         copyMatrix(tempOutput,outputMatrixs.at(layer));
     }

    return  tempOutput;
}


void NueralNetwork::updateWeights(Matrix &in, Matrix &actual) {
    double h = 0.001;
    for(Matrix& weightMatrix: weightMatrixs){
        for(int i = 0; i < weightMatrix.numRows * weightMatrix.numColumns; i++){
            Matrix initialCost =  mseLoss(feedFoward(in),actual);
            weightMatrix.setRawElement(i,weightMatrix.getRawElement(i) + h);

            Matrix afterNudge = mseLoss(feedFoward(in),actual);
            weightMatrix.setRawElement(i,weightMatrix.getRawElement(i) - h);
            Matrix derivitive = minusMatrix(afterNudge,initialCost);

            weightMatrix.setRawElement(i,(weightMatrix.getRawElement(i) - (derivitive.getRawElement(0) / h) * 0.1));
        }
    }

}

/**
 * updates gradientMatrixs with calculations from new cost.
 * */
void NueralNetwork::backprop(const Matrix &cost){

}
