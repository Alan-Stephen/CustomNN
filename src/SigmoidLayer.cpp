
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
    return out;
}

void SigmoidLayer::randomizeParams() {
    return;
}

Matrix SigmoidLayer::getDerivitive(const Matrix &in) {
   Matrix out(in.numRows,in.numCols);
    for (int i = 0; i < in.size(); ++i) {
        // todo : to the most obvious optimistation ever
        double value = sigmoid(in.getRawElement(i)) * (1 - sigmoid(in.getRawElement(i)));
        out.setRawElement(i,value);
    }
}

double SigmoidLayer::sigmoid(double in) const {
    return (1 + exp(in * -1))
}

void SigmoidLayer::printLayer() const {
    return;
}

