//
// Created by alan on 14/06/23.
//

#ifndef CUSTOMNN_SIGMOIDLAYER_H
#define CUSTOMNN_SIGMOIDLAYER_H

#include "matrix.h"
#include "Layer.h"

class SigmoidLayer : public Layer {
public:
    SigmoidLayer(int size);
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix feedBackward(const Matrix &error) override;
    void printLayer() const override;

    Matrix getDerivitive(const Matrix &in);

    int getIn() const override;
    int getOut() const override;
private:
    int _size;
    double sigmoid(double in) const;
    Matrix _output;
    Matrix _previousLayerOutputs;
};

#endif //CUSTOMNN_SIGMOIDLAYER_H
