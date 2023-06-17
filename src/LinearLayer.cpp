#include "../include/matrix.h"
#include "../include/LinearLayer.h"

//
// Created by alan on 11/06/23.
//
Matrix &LinearLayer::getWeightMatrix() {
    return _weightMatrix;
}

Matrix &LinearLayer::getBiasMatrix() {
    return _biasMatrix;
}

LinearLayer::LinearLayer(int in, int out, PyNet &network,int layerNum): _biasMatrix(out,1), _weightMatrix(out,in), _output(out,1),
                                           _weightGradientMatrix(out,in), _biasGradientMatrix(out,1), _net(network), _layerNum(layerNum){}

Matrix LinearLayer::feedForward(const Matrix& in) {
    Matrix temp = multiplyMatrix(_weightMatrix,in);

    if(!isSameDimensions(temp,_output)){
        std::cout << "[ERROR] Dimensions for copying to output matrixes during feed foward not correct";
        std::cout << temp.numRows << " " << temp.numCols << "\n";
        std::cout << _output.numRows << " " << _output.numCols << "\n";
        exit(1);
    }

    for(int i = 0; i < temp.numCols * temp.numRows; i++){
        double value = temp.getRawElement(i);
        _output.setRawElement(i,value);
    }
    _output = addMatrix(temp,_biasMatrix);
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

void LinearLayer::updateGradients(Matrix &error, Matrix &previousLayerOutputs) {
   /*
    * how to update weight matrixes, D'output/D'weight = activations from previous layer
    * */
   previousLayerOutputs.transpose();
   _weightGradientMatrix = multiplyMatrix(0.001,multiplyMatrix(error,previousLayerOutputs));
    _biasGradientMatrix = multiplyMatrix(0.001,error);
    previousLayerOutputs.transpose();
    // update bias gradients
}

Matrix LinearLayer::layerOutput() {
    return _output;
}

void LinearLayer::applyGradients() {
    _weightMatrix.minus(_weightGradientMatrix);
    _biasMatrix.minus(_biasGradientMatrix);
}

void LinearLayer::clearGradients() {
    for(int i = 0; i < _weightGradientMatrix.numRows * _weightGradientMatrix.numCols; i++){
        _weightGradientMatrix.setRawElement(i,0);
    }

    for(int i = 0; i < _biasGradientMatrix.numRows * _biasGradientMatrix.numCols; i++){
        _biasGradientMatrix.setRawElement(i,0);
    }
}



