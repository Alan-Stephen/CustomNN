
#include "../include/SigmoidLayer.h"
#include <cmath>

void SigmoidLayer::updateGradients(Matrix &error, Matrix &previousLayerActivations) {
    return;
}

void SigmoidLayer::applyGradients() {
    return;
}

void SigmoidLayer::clearGradients() {
    clear(_output);
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
    _output.add(out);
    return out;
}

void SigmoidLayer::randomizeParams() {
    return;
}

Matrix SigmoidLayer::getDerivitive(const Matrix &in) {
    // todo : make this more cache friendly
   Matrix out(_output.numRows,_output.numCols);
    for (int i = 0; i < _output.size();++i) {
        double value = sigmoid(_output.getRawElement(i));
        // todo : to the most obvious optimistation ever
        double deriv = value  * (1 - value);
        out.setRawElement(i, deriv);
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

SigmoidLayer::SigmoidLayer(int size): _in(size), _out(size), _output(size,1) {}

int SigmoidLayer::getIn() const {
   return _in;
}

int SigmoidLayer::getOut() const {
    return _out;
}

Matrix SigmoidLayer::feedBackward(const Matrix &error) {
    return hadamardProduct(error, getDerivitive(_output));
}

