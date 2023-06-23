//
// Created by alan on 22/06/23.
//

#ifndef CUSTOMNN_RELULAYER_H
#define CUSTOMNN_RELULAYER_H

#include "matrix.h"
#include "Layer.h"

class ReLuLayer : public Layer {
public:
    ReLuLayer(size_t size, double leakValue = 0);

    int getIn() const override;
    int getOut() const override;
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix feedBackward(const Matrix &error) override;
    void printLayer() const override;
private:
    size_t _size;
    Matrix _output;
    double _leakValue;
};

#endif //CUSTOMNN_RELULAYER_H
