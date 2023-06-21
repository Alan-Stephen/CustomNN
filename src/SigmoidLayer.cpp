
#include "../include/SigmoidLayer.h"
#include <cmath>

void SigmoidLayer::updateGradients(Matrix &error, Matrix &previousLayerActivations) {
    return;
}

void SigmoidLayer::applyGradients() {
    return;
}

void SigmoidLayer::clearGradients() {
    return;
}

Matrix SigmoidLayer::layerOutput() {
    return _output;
}

Matrix SigmoidLayer::feedForward(const Matrix &in) {
    Matrix out(in.numRows, in.numCols);
    for (int i = 0; i < in.size(); ++i) {
        double value = sigmoid(in.getRawElement(i));
        out.setRawElement(i,value);
    }
    _output = out;
    return out;
}

void SigmoidLayer::randomizeParams() {
    return;
}

Matrix SigmoidLayer::getDerivitive(const Matrix &in) {
   Matrix out(_output.numRows,_output.numCols);
    for (int i = 0; i < _output.size();++i) {
        // todo : to the most obvious optimistation ever
        double value = sigmoid(_output.getRawElement(i)) * (1 - sigmoid(_output.getRawElement(i)));
        out.setRawElement(i,value);
    }
    return out;
}

double SigmoidLayer::sigmoid(double in) const {
    double temp1 = in * -1.0;
    temp1 = exp(temp1);
    temp1 = 1.0 + temp1;
    temp1 = 1 / temp1;
    return temp1;
}

void SigmoidLayer::printLayer() const {
    return;
}

SigmoidLayer::SigmoidLayer(int size): _in(size), _out(size) {}

int SigmoidLayer::getIn() const {
   return _in;
}

int SigmoidLayer::getOut() const {
    return _out;
}

Matrix SigmoidLayer::feedBackward(const Matrix &error) {
    return hadamardProduct(error, getDerivitive(_output));
}

