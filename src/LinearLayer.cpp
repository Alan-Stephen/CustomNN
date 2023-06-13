#include "../include/matrix.h"
#include "../include/LinearLayer.h"

//
// Created by alan on 11/06/23.
//
const Matrix &LinearLayer::getWeightMatrix() const {
    return _weightMatrix;
}

const Matrix &LinearLayer::getBiasMatrix() const {
    return _biasMatrix;
}

LinearLayer::LinearLayer(int in, int out, PyNet &network,int layerNum): _biasMatrix(out,1), _weightMatrix(out,in), _output(out,1),
                                           _weightGradientMatrix(out,in), _biasGradientMatrix(out,1), _net(network), _layerNum(layerNum){}

Matrix LinearLayer::feedForward(const Matrix& in) {
    Matrix temp = multiplyMatrix(_weightMatrix,in);

    if(!isSameDimensions(temp,_output)){
        std::cout << "[ERROR] Dimensions for copying to output matrixes during feed foward not correct";
        std::cout << temp.numRows << " " << temp.numColumns << "\n";
        std::cout << _output.numRows << " " << _output.numColumns << "\n";
        exit(1);
    }

    for(int i = 0; i < temp.numColumns * temp.numRows;i++){
        double value = temp.getRawElement(i);
        _output.setRawElement(i,value);
    }
    return addMatrix(temp,_biasMatrix);

}

void LinearLayer::randomiseParams() {
    this->randomiseWeights();
    this->randomiseBiases();
}

void LinearLayer::randomiseWeights() {
    randomizeMatrix(_weightMatrix);
}

void LinearLayer::randomiseBiases()  {
    randomizeMatrix(_biasMatrix);
}

void LinearLayer::updateGradients(const Matrix &error, const Matrix &previousLayerOutputs) {
   /*
    * how to update weight matrixes, D'output/D'weight = activations from previous layer
    * */
    _weightGradientMatrix = multiplyMatrix(0.05,multiplyMatrix(previousLayerOutputs,error));
    _biasGradientMatrix = multiplyMatrix(0.05,error);
    previousLayerOutputs.printMatrix();
    _weightGradientMatrix.printMatrix();
    _biasGradientMatrix.printMatrix();
    _weightMatrix.printMatrix();
    _biasMatrix.printMatrix();
    // update bias gradients
}

Matrix LinearLayer::layerOutput() {
    return _output;
}

void LinearLayer::applyGradients() {
    _weightMatrix.minus(_weightGradientMatrix);
}

void LinearLayer::clearGradients() {
    for(int i = 0; i < _weightGradientMatrix.numRows * _weightGradientMatrix.numColumns;i++){
        _weightGradientMatrix.setRawElement(i,0);
    }

    for(int i = 0; i < _biasGradientMatrix.numRows * _biasGradientMatrix.numColumns;i++){
        _biasGradientMatrix.setRawElement(i,0);
    }
}



