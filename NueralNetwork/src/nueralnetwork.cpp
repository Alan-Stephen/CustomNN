#include "nueralnetwork.h"

NueralNetwork::NueralNetwork(const int* structure, int depth) {
        const int* prev;
        weightMatrix.reserve(depth);
        biasMatrix.reserve(depth);
        for(int i = 0; i < depth; i++) {
            if(i == 0){
                prev = structure;
                continue;
            }

            biasMatrix.push_back(RandomMatrix(structure[i],1));

            weightMatrix.push_back(RandomMatrix(structure[i],*prev));
            std::cout << " i :" << structure[i] << " prevv : " << *prev << std::endl;
            prev++;
        }
        for(int i = 0; i < weightMatrix.size();i++) {
            weightMatrix[i].printMatrix();
            biasMatrix[i].printMatrix();
        }

}

double NueralNetwork::activate(double x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

Matrix NueralNetwork::feedFoward(Matrix a){
}