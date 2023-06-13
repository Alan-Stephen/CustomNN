//
// Created by alan on 11/06/23.
//

#ifndef CUSTOMNN_LINEARLAYER_H
#define CUSTOMNN_LINEARLAYER_H

#include "matrix.h"
#include "py_net.h"
class LinearLayer {
public:
    LinearLayer(int in,int out, PyNet &net,int layerNum);

    [[nodiscard]] const Matrix &getWeightMatrix() const;
    [[nodiscard]] const Matrix &getBiasMatrix() const;

    void randomiseWeights();
    void randomiseBiases();
    void randomiseParams();

    /**
     * Updates the output values for each layer as well.
     * */
    [[nodiscard]] Matrix feedForward(const Matrix& in);

    void updateGradients(const Matrix &error,const Matrix &previousLayerActivations);
    void applyGradients();
    Matrix layerOutput();

    void clearGradients();

private:
    int _layerNum;

    Matrix _output;
    PyNet &_net;

    Matrix _weightGradientMatrix;
    Matrix _biasGradientMatrix;
    Matrix _weightMatrix;
    Matrix _biasMatrix;

};

#endif //CUSTOMNN_LINEARLAYER_H
