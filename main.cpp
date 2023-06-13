#include <iostream>
#include <vector>
#include "include/matrix.h"
#include "include/nueral_network.h"
#include "include/LinearLayer.h"
#include "include/py_net.h"

// todo: do a lot of refactoring
double functionToEstimate(const double in){
    return in * 2;
}


int main(int argc, char const *argv[]){
    Matrix in = Matrix(1,1);
    Matrix actual = Matrix(1,1);

    randomizeMatrix(in);
    for(int i = 0; i < actual.numColumns * actual.numRows; i++){
        double value = in.getRawElement(i) * 2;
        actual.setRawElement(i,value);
    }
    std::vector<int> structure = {1,1};

    PyNet net = PyNet(structure);

    net.randomiseParams();

    Matrix out = net.feedFoward(in);

    std::cout << "out\n";
    out.printMatrix();

    std::cout << "actual\n";
    actual.printMatrix();

    Matrix loss = mseLossDerivitive(actual,out);
    std::cout << "in and rest\n";
    net.updateGradients(loss,in);

}