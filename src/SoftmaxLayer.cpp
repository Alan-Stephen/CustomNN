//
// Created by alan on 26/06/23.
//
#include <valarray>
#include "../include/SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(size_t size): _size(size) {}

int SoftmaxLayer::getIn() const {
    return _size;
}

int SoftmaxLayer::getOut() const {
    return _size;
}

void SoftmaxLayer::updateGradients(Matrix &error, Matrix &previousLayerActivations) {
    return;
}

void SoftmaxLayer::applyGradients() {
    return;
}

void SoftmaxLayer::clearGradients() {
    return;
}

// todo : optimisation, return reference instead of copying
Matrix SoftmaxLayer::layerOutput() {
    return _output;
}

Matrix SoftmaxLayer::feedForward(const Matrix &in) {
    Matrix out(in.numRows,in.numCols);
    double total = 0.0;

    for (double value: in.data) {
        total += exp(value);
    }

    for(int i = 0; i < in.size(); i++){
        double value = exp(in.getRawElement(i)) / total;
        out.setRawElement(i, value);
    }
    _output = out;

    return out;
}

void SoftmaxLayer::randomizeParams() {
    return;
}

Matrix SoftmaxLayer::feedBackward(const Matrix &error) {

    Matrix deriv(_output.numRows, _output.numRows);
    for (int i = 0; i < _output.numRows; ++i) {
        for (int j = 0; j < _output.numRows; ++j) {
            double value = _output.getRawElement(i) * ((i == j) - _output.getRawElement(j));
            deriv.setElement(i, j, value);
        }
    }
    // deriv will be n*n
    return multiplyMatrix(deriv,error);
}

void SoftmaxLayer::printLayer() const {
    return;
}
