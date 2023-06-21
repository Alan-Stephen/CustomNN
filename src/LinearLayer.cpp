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

LinearLayer::LinearLayer(int in, int out,double learningRate): _learningRate(learningRate),_biasMatrix(out,1), _weightMatrix(out,in), _output(out,1),
                                           _weightGradientMatrix(out,in), _biasGradientMatrix(out,1){}

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
   _weightGradientMatrix = multiplyMatrix(error,previousLayerOutputs);
    _biasGradientMatrix = error;
    previousLayerOutputs.transpose();
    // update bias gradients
}

Matrix LinearLayer::layerOutput() {
    return _output;
}

void LinearLayer::applyGradients() {
    _weightMatrix.minus(multiplyMatrix(_learningRate,_weightGradientMatrix));
    _biasMatrix.minus(multiplyMatrix(_learningRate,_biasGradientMatrix));
}

void LinearLayer::clearGradients() {
    for(int i = 0; i < _weightGradientMatrix.numRows * _weightGradientMatrix.numCols; i++){
        _weightGradientMatrix.setRawElement(i,0);
    }

    for(int i = 0; i < _biasGradientMatrix.numRows * _biasGradientMatrix.numCols; i++){
        _biasGradientMatrix.setRawElement(i,0);
    }
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



