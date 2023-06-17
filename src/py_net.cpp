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

void PyNet::printLayers() {
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
    Matrix tempError = error;
    Matrix tempInput = input;
    for(int layer = layers.size() - 1; layer >= 0; layer--){
       LinearLayer &currentLayer = layers.at(layer);
       if(layer == 0){
           currentLayer.updateGradients(tempError, tempInput);
           return;
       }
       else {
           Matrix layerOutputCopy = getLayerOutputs(layer - 1);
           currentLayer.updateGradients(tempError,layerOutputCopy);
       }

       // update errors
        Matrix &weights = currentLayer.getWeightMatrix();
        weights.transpose();
        tempError = multiplyMatrix(weights,tempError);
        weights.transpose();
    }
}

Matrix PyNet::getLayerOutputs(int layer) {
    return layers.at(layer).layerOutput();
}

LinearLayer &PyNet::getLayer(int layer) {
   return layers.at(0);
}

void PyNet::applyGradients() {
    for(LinearLayer &layer: layers)
        layer.applyGradients();
}

void PyNet::clearGradients() {
    for(LinearLayer &layer: layers)
        layer.clearGradients();
}

