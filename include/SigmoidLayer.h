//
// Created by alan on 14/06/23.
//

#ifndef CUSTOMNN_SIGMOIDLAYER_H
#define CUSTOMNN_SIGMOIDLAYER_H

#include "matrix.h"
#include "Layer.h"

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(int in, int out);
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix getDerivitive(const Matrix &in) override;
    void printLayer() const override;

    int getIn() const override;
    int getOut() const override;
private:
    int _in;
    int _out;
    double sigmoid(double in) const;
    Matrix _output;
};

#endif //CUSTOMNN_SIGMOIDLAYER_H
