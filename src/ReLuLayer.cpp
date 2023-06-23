

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
        double value = std::max(_leakValue * in.getRawElement(i),in.getRawElement(i));
        out.setRawElement(i,value);
    }

    _output = out;
    return out;
}

Matrix ReLuLayer::feedBackward(const Matrix &error) {
    Matrix out(error.numRows,error.numCols);

    for (int i = 0; i < error.size(); ++i) {
        double value;
        if(error.getRawElement(i) <= 0)
            value = error.getRawElement(i) * _leakValue;
        else
            value = error.getRawElement(i);
        out.setRawElement(i,value);
    }

    return out;
}

void ReLuLayer::printLayer() const {
}

ReLuLayer::ReLuLayer(size_t size, double leakValue): _size(size), _leakValue(leakValue) {}