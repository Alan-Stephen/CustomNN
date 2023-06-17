//
// Created by alan on 12/06/23.
//

#ifndef CUSTOMNN_PY_NET_H
#define CUSTOMNN_PY_NET_H

#include <vector>
#include "matrix.h"

/**
 *
 * alternative nueral network, a nueral network is just a array of layers.
 * */
class LinearLayer;

class PyNet {
public:


    PyNet(std::vector<int> structure);
    Matrix getLayerOutputs(int layer);
    LinearLayer &getLayer(int layer);
    void printLayers();
    void randomiseParams();

    Matrix feedFoward(const Matrix &in);
    void updateGradients(const Matrix &error, const Matrix &input);
    void applyGradients();
    void clearGradients();

private:

    std::vector<LinearLayer> layers;
};



#endif //CUSTOMNN_PY_NET_H
