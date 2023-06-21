//
// Created by alan on 12/06/23.
//
#include "../include/py_net.h"
#include "../include/LinearLayer.h"

PyNet::PyNet() = default;


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

       tempError = _layers.at(layer)->feedBackward(tempError);
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

void PyNet::addLayer(Layer *layer) {
    if ((_layers.size() != 0) && (_layers.back()->getOut() != layer->getIn())){
        std::cout << "[ERROR] Invalid Layer sizes";
        exit(1);
    }
    _layers.push_back(std::unique_ptr<Layer>(layer));
}

