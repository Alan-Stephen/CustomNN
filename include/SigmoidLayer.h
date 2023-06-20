//
// Created by alan on 14/06/23.
//

#ifndef CUSTOMNN_SIGMOIDLAYER_H
#define CUSTOMNN_SIGMOIDLAYER_H

#include "matrix.h"
#include "Layer.h"

class SigmoidLayer : public Layer {
public:
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix getDerivitive(const Matrix &in) override;
    void printLayer() const override;
private:
    double sigmoid(double in) const;
    Matrix _output;
};

#endif //CUSTOMNN_SIGMOIDLAYER_H
