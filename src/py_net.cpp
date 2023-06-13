//
// Created by alan on 12/06/23.
//
#include "../include/py_net.h"
#include "../include/LinearLayer.h"

PyNet::PyNet(std::vector<int> structure) {
    layers.reserve(structure.size());

    int prev = structure.at(0);
    for(int i = 1; i < structure.size(); i++){
        layers.emplace_back(prev,structure.at(i), *this,i);
        prev = structure.at(i);
    }
}

void PyNet::printLayers() const {
    for(int i = 0; i < layers.size(); i++){
        std::cout << "LAYER : "<< i <<"\n";
        layers.at(i).getWeightMatrix().printMatrix();
        std::cout << "LAYER: " << i << "\n";
        layers.at(i).getBiasMatrix().printMatrix();
    }
}

Matrix PyNet::feedFoward(const Matrix &in) {
    Matrix temp = layers.at(0).feedForward(in);

    for(int i = 1; i < layers.size(); i++){
        temp = layers.at(i).feedForward(temp);
    }
    return temp;
}

void PyNet::randomiseParams() {
    for(int i = 0; i < layers.size(); i++){
        layers.at(i).randomiseParams();
    }
}

void PyNet::updateGradients(const Matrix &error,const Matrix &input) {
    layers.at(0).updateGradients(error,input);
}

Matrix PyNet::getLayerOutputs(int layer) {
    return layers.at(layer).layerOutput();
}

