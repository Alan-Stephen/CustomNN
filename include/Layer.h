//
// Created by alan on 14/06/23.
//

#ifndef CUSTOMNN_LAYER_H
#define CUSTOMNN_LAYER_H

#include "matrix.h"

class Layer {
public:
    virtual int getIn() const = 0;
    virtual int getOut() const = 0;
    virtual void updateGradients(Matrix &error, Matrix &previousLayerActivations) = 0;
    virtual void applyGradients() = 0;
    virtual void clearGradients() = 0;
    virtual Matrix layerOutput() = 0;
    virtual Matrix feedForward(const Matrix &in) = 0;
    virtual void randomizeParams() = 0;
    virtual Matrix feedBackward(const Matrix &error) = 0;
    virtual void printLayer() const = 0;
};
#endif //CUSTOMNN_LAYER_H
