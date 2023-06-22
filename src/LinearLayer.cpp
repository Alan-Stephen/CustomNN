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

LinearLayer::LinearLayer(int in, int out,double learningRate, int batchSize): _batchSize(batchSize), _learningRate(learningRate),_biasMatrix(out,1), _weightMatrix(out,in), _output(out,1),
                                           _weightGradientMatrix(out,in), _biasGradientMatrix(out,1){}

Matrix LinearLayer::feedForward(const Matrix& in) {
    Matrix temp = multiplyMatrix(_weightMatrix,in);

    if(!isSameDimensions(temp,_output)){
        std::cout << "[ERROR] Dimensions for copying to output matrixes during feed foward not correct";
        std::cout << temp.numRows << " " << temp.numCols << "\n";
        std::cout << _output.numRows << " " << _output.numCols << "\n";
        exit(1);
    }
    temp = addMatrix(temp,_biasMatrix);
    _output = temp;
    return temp;
}

void LinearLayer::randomizeParams() {
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
   _weightGradientMatrix.add(multiplyMatrix(error,previousLayerOutputs));
    _biasGradientMatrix.add(error);
    previousLayerOutputs.transpose();
    // update bias gradients
}

Matrix LinearLayer::layerOutput() {
    return _output;
}

void LinearLayer::applyGradients() {
    _weightMatrix.minus(multiplyMatrix(_learningRate / _batchSize,_weightGradientMatrix));
    _biasMatrix.minus(multiplyMatrix(_learningRate / _batchSize,_biasGradientMatrix));
}

void LinearLayer::clearGradients() {
    clear(_weightGradientMatrix);
    clear(_biasGradientMatrix);
}

void LinearLayer::printLayer() const {
    // todo : finish this shit
}

int LinearLayer::getIn() const {
    return _weightMatrix.numCols;
}

int LinearLayer::getOut() const {
    return _weightMatrix.numRows;
}

// takes in error passes it through layer.
Matrix LinearLayer::feedBackward(const Matrix &error) {
    _weightMatrix.transpose();
    Matrix temp =  multiplyMatrix(_weightMatrix,error);
    _weightMatrix.transpose();
    return temp;
}



