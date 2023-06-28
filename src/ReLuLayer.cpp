

#include "../include/ReLuLayer.h"

int ReLuLayer::getIn() const {
    return _size;
}

int ReLuLayer::getOut() const {
    return _size;
}

void ReLuLayer::updateGradients(Matrix &error, Matrix &previousLayerActivations) {
    return;
}

void ReLuLayer::applyGradients() {
    return;
}

void ReLuLayer::clearGradients() {
    return;
}

Matrix ReLuLayer::layerOutput() {
    return _output;
}

void ReLuLayer::randomizeParams() {
    return;
}

Matrix ReLuLayer::feedForward(const Matrix &in) {
    Matrix out(in.numRows,in.numCols);

    for (int i = 0; i < out.size(); ++i) {
        double value = in.getRawElement(i);
        if(value < 0)
            value = value * _leakValue;
        out.setRawElement(i,value);
    }

    _output = out;
    _previousLayerOutputs = in;
    return out;
}

Matrix ReLuLayer::feedBackward(const Matrix &error) {
    Matrix out(_output.numRows, _output.numCols);

    for (int i = 0; i < _previousLayerOutputs.size(); ++i) {
        double value;
        if(_previousLayerOutputs.getRawElement(i) < 0)
            value = _leakValue;
        else
            value = 1.0;
        out.setRawElement(i,value);
    }

    return hadamardProduct(error,out);
}

void ReLuLayer::printLayer() const {
}

ReLuLayer::ReLuLayer(size_t size, double leakValue): _size(size), _leakValue(leakValue) {}