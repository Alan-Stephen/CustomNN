//
// Created by alan on 26/06/23.
//

#ifndef CUSTOMNN_SOFTMAXLAYER_H
#define CUSTOMNN_SOFTMAXLAYER_H

#include "Layer.h"

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(size_t size);
    int getIn() const override;
    virtual int getOut() const override;
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix feedBackward(const Matrix &_output) override;
    void printLayer() const override;
private:
    Matrix _output;
    size_t _size;
    Matrix _previousLayerOutputs;
};


#endif //CUSTOMNN_SOFTMAXLAYER_H
