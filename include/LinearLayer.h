//
// Created by alan on 11/06/23.
//

#ifndef CUSTOMNN_LINEARLAYER_H
#define CUSTOMNN_LINEARLAYER_H

#include "matrix.h"
#include "py_net.h"
#include "Layer.h"

class LinearLayer : public Layer{
public:
    LinearLayer(int in,int out);

    [[nodiscard]] Matrix &getWeightMatrix();
    [[nodiscard]] Matrix &getBiasMatrix();

    void randomiseWeights();
    void randomiseBiases();
    void randomizeParams() override;

    /**
     * Updates the output values for each layer as well.
     * */
    [[nodiscard]] Matrix feedForward(const Matrix& in) override;

    void updateGradients(Matrix &error,Matrix &previousLayerActivations) override;
    void applyGradients() override;
    Matrix layerOutput() override;
    void clearGradients() override;

    Matrix getDerivitive(const Matrix &in) override;

    void printLayer() const override;

private:

    Matrix _output;

    Matrix _weightGradientMatrix;
    Matrix _biasGradientMatrix;
    Matrix _weightMatrix;
    Matrix _biasMatrix;

};

#endif //CUSTOMNN_LINEARLAYER_H
