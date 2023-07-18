//
// Created by alan on 29/06/23.
//

#ifndef CUSTOMNN_CONV2DLAYER_H
#define CUSTOMNN_CONV2DLAYER_H

#include "matrix.h"
#include "Layer.h"

class Conv2DLayer : public Layer {
public:
    friend class Tester;
    Conv2DLayer(std::pair<uint32_t, uint32_t> inDims, std::pair<uint32_t, uint32_t> kernelDims, uint32_t numKernels,
                uint32_t padding, std::pair<uint32_t, uint32_t> strideDims, double learningRate, int batchSize);
    int getIn() const override;
    int getOut() const override;
    void updateGradients(Matrix &error, Matrix &previousLayerActivations) override;
    void applyGradients() override;
    void clearGradients() override;
    Matrix layerOutput() override;
    Matrix feedForward(const Matrix &in) override;
    void randomizeParams() override;
    Matrix feedBackward(const Matrix &error) override;
    void printLayer() const override;
private:
    Matrix applyKernel(const Matrix &in, const Matrix &kernel);
    std::pair<uint32_t,uint32_t> _inDims;
    std::pair<uint32_t,uint32_t> _outDims;
    const std::pair<uint32_t,uint32_t> _stride;
    const size_t _padding;

    std::vector<double> _biases;
    std::vector<Matrix> _kernels;

    std::vector<double> _biasGradients;
    std::vector<Matrix> _kernelGradients;

    Matrix _out;
    Matrix _previousLayerActivations;

    double _learningRate;
    int _batchSize;
};

#endif //CUSTOMNN_CONV2DLAYER_H
