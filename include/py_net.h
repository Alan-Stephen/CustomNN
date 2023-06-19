//
// Created by alan on 12/06/23.
//

#ifndef CUSTOMNN_PY_NET_H
#define CUSTOMNN_PY_NET_H

#include <vector>
#include "matrix.h"
#include "Layer.h"

/**
 *
 * alternative nueral network, a nueral network is just a array of _layers.
 * */
class Layer;

class PyNet {
public:


    PyNet(std::vector<int> structure);
    Matrix getLayerOutputs(int layer);
    void printLayers();
    void randomiseParams();

    Matrix feedFoward(const Matrix &in);
    void updateGradients(const Matrix &error, const Matrix &input);
    void applyGradients();
    void clearGradients();

private:

    std::vector<std::unique_ptr<Layer>> _layers;
};



#endif //CUSTOMNN_PY_NET_H
