//
// Created by alan on 12/06/23.
//
#include "../include/py_net.h"
#include "../include/LinearLayer.h"

PyNet::PyNet(std::vector<int> structure) {
    _layers.reserve(structure.size());

    int prev = structure.at(0);
    for(int i = 1; i < structure.size(); i++){
        _layers.push_back(std::make_unique<LinearLayer>(prev,structure.at(i)));
        prev = structure.at(i);
    }
}

void PyNet::printLayers() {
    // finish this
    for(int i = 0; i < _layers.size(); i++){
        std::cout << "LAYER : "<< i <<"\n";
        std::cout << "LAYER: " << i << "\n";
    }
}

Matrix PyNet::feedFoward(const Matrix &in) {
    Matrix temp = _layers.at(0)->feedForward(in);

    for(int i = 1; i < _layers.size(); i++){
        temp = _layers.at(i)->feedForward(temp);
    }
    return temp;
}

void PyNet::randomiseParams() {
    for(int i = 0; i < _layers.size(); i++){
        _layers.at(i)->randomizeParams();
    }
}

void PyNet::updateGradients(const Matrix &error,const Matrix &input) {
    Matrix tempError = error;
    Matrix tempInput = input;
    for(int layer = _layers.size() - 1; layer >= 0; layer--){
       if(layer == 0){
           _layers.at(layer)->updateGradients(tempError, tempInput);
           return;
       }
       else {
           Matrix layerOutputCopy = getLayerOutputs(layer - 1);
           _layers.at(layer)->updateGradients(tempError,layerOutputCopy);
       }

       // update errors
        Matrix temp(1,1);
        Matrix weights = _layers.at(layer)->getDerivitive(temp);
        weights.transpose();
        tempError = multiplyMatrix(weights,tempError);
        weights.transpose();
    }
}

Matrix PyNet::getLayerOutputs(int layer) {
    return _layers.at(layer)->layerOutput();
}


void PyNet::applyGradients() {
    for(std::unique_ptr<Layer> &layer: _layers)
        layer->applyGradients();
}

void PyNet::clearGradients() {
    for(std::unique_ptr<Layer> &layer: _layers)
        layer->clearGradients();
}

