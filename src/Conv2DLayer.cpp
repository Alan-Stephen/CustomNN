//
// Created by alan on 01/07/23.
//

#include <cmath>
#include "../include/Conv2DLayer.h"


/**
 * std::pair(height,width)
 *           row , col
 *
 * */
 // todo : emplace values into _out
Conv2DLayer::Conv2DLayer(std::pair<uint32_t, uint32_t> inDims, std::pair<uint32_t, uint32_t> kernelDims,
                         uint32_t numKernels, uint32_t padding, std::pair<uint32_t, uint32_t> strideDims,
                         double learningRate, int batchSize)
        : _stride(strideDims), _padding(padding), _inDims(inDims), _learningRate(learningRate), _batchSize(batchSize) {
    uint32_t outHeight = 1 + (double) (inDims.first + 2 * padding - kernelDims.first) / (double) strideDims.first;
    double checkHeight  = 1 + (double) (inDims.first + 2 * padding - kernelDims.first) / (double) strideDims.first;
    if(outHeight != checkHeight){
        std::cout << "invalid size of kernel / padding / stride / height for generating conv layer \n";
        exit(1);
    }

    uint32_t outWidth = 1 + (double) (inDims.second + 2 * padding - kernelDims.second) / (double) strideDims.second;
    double checkWidth  = 1 + (double) (inDims.second + 2 * padding - kernelDims.second) / (double) strideDims.second;
    if(outWidth != checkWidth){
        std::cout << "invalid size of kernel / padding / stride / height for generating conv layer \n";
        exit(1);
    }

    for (int i = 0; i < numKernels; ++i) {
        _kernelGradients.emplace_back(kernelDims.first * kernelDims.second,1);
        _kernels.emplace_back(kernelDims.first, kernelDims.second);
        _biases.emplace_back(0.0);
    }

    _outDims = std::pair<uint32_t,uint32_t>(outHeight, outWidth);

    _out = Matrix(outHeight*outWidth,1);
}

int Conv2DLayer::getIn() const {
    return _inDims.first * _inDims.second;
}

int Conv2DLayer::getOut() const {
    return _outDims.first * _outDims.second * _kernels.size();
}

void Conv2DLayer::updateGradients(Matrix &error, Matrix &previousLayerActivations) {
    // complete redo
    // first translate error into correct size matrix for each kernel out
    // then construct jacobian matirx by looping through row and column of jacobian matrix and filling
    // in appropriate values accounting for the stride
    //

    for (int kernel = 0; kernel < _kernels.size(); ++kernel) {
        Matrix jacobian(_kernels.at(kernel).size(),((double) getOut() / (double) _kernels.size()));
        Matrix localError = Matrix((double) error.numRows / (double) _kernels.size(), 1);

        const int lastWritten = (kernel) * localError.size();
        for (int i = 0; i < localError.size(); ++i) {
            double value = error.getRawElement(i + lastWritten);
            localError.setRawElement(i,value);
        }

        for (int kernelElement = 0; kernelElement < jacobian.numRows; ++kernelElement) {

            int kernelRow = floor((double) kernelElement / (double) _kernels.at(kernel).numCols);
            int kernelCols = kernelElement % _kernels.at(kernel).numCols;

            for (int outputElement = 0; outputElement < jacobian.numCols; ++outputElement) {
                int outputRow = floor((double) outputElement / (double) _outDims.second);
                int outputCols = (outputElement % (int) _outDims.second);

                int index = (outputRow + kernelRow) * static_cast<int>(_outDims.second) + (outputCols + kernelCols);
                double value = previousLayerActivations.getRawElement(index);

                jacobian.setElement(outputElement,kernelElement,value);
            }
        }

        _kernelGradients.at(kernel).add(multiplyMatrix(jacobian,localError));
    }
}

void Conv2DLayer::applyGradients() {
    for (int kernel = 0; kernel < _kernels.size(); ++kernel) {
        Matrix &currKernel = _kernels.at(kernel);

        for (int i = 0; i < currKernel.size(); ++i) {
            double value = currKernel.getRawElement(i) - _kernelGradients.at(kernel).getRawElement(i) *
                    (_learningRate / static_cast<double>(_batchSize));
            currKernel.setRawElement(i,value);
        }
    }

    // implement bias gradients
}

void Conv2DLayer::clearGradients() {
    for(Matrix &kernel: _kernelGradients)
        clear(kernel);
    for(double &value: _biasGradients)
        value = 0;
}

Matrix Conv2DLayer::layerOutput() {
    return _out;
}

Matrix Conv2DLayer::feedForward(const Matrix &in) {
    Matrix out(_outDims.first * _outDims.second * _kernels.size(),1);
    int last_written = 0;
    for(int kernel = 0; kernel < _kernels.size(); kernel++){
        Matrix currOut = applyKernel(in,_kernels.at(kernel));
        for (int i = 0; i < currOut.size(); ++i) {
            double value = currOut.getRawElement(i);
            out.setRawElement(i + last_written, value);
        }

        last_written += currOut.size();
    }
    _out = out;
    return out;
}

void Conv2DLayer::randomizeParams() {
    for(Matrix &kernel: _kernels)
        randomizeMatrix(kernel);
    for(double &bias: _biases)
       bias = ((double) rand() / (RAND_MAX>>1)) - 1;
}

Matrix Conv2DLayer::feedBackward(const Matrix &error) {
    return Matrix();
}

void Conv2DLayer::printLayer() const {
}

Matrix Conv2DLayer::applyKernel(const Matrix &in, const Matrix &kernel) {
    Matrix out(_outDims.first ,_outDims.second);
    for (int m_row = 0; m_row < out.numRows; ++m_row) {
        for (int m_col = 0; m_col < out.numCols; ++m_col) {
            double value = 0;
            for (int k_row = 0; k_row < _kernels.at(0).numRows; ++k_row) {
                for (int k_col = 0; k_col < _kernels.at(0).numCols; ++k_col) {
                    uint32_t xCol = m_col * _stride.second + k_col;
                    uint32_t xRow = m_row * _stride.first + k_row;

                    double xVal;
                    if(xCol > in.numCols || xRow > in.numRows) {
                        std::cout << "OUT OF BOUNDS\n";
                        xVal = 0;
                    }
                    else
                        xVal = in.getElement(xCol,xRow);

                    value += xVal * kernel.getElement(k_col,k_row);
                }
            }
            out.setElement(m_row,m_col,value);
        }
    }
    _out = out;
    return out;
}
