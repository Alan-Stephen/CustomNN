#include <iostream>
#include <vector>
#include "include/matrix.h"
#include "include/nueral_network.h"
#include "include/LinearLayer.h"
#include "include/py_net.h"
#include "include/MNISTLabel.h"

// todo: do a lot of refactoring
double functionToEstimate(const double in){
    return in * 2;
}


int main(int argc, char const *argv[]){

    std::vector<MNISTLabel> data = parseMNISTLabels("../data/mnist_test.csv");



    std::vector<int> structure = {784,100,10};
    PyNet net = PyNet(structure);
    net.randomiseParams();

    net.feedFoward(data.at(0)._in).printMatrix();

#if 0
    Matrix test = Matrix(10,10);
    randomizeMatrix(test);

    std::cout << test.data.size();

    const int NUM_TESTS = 10;
    std::vector<Matrix> ins;
    std::vector<Matrix> actuals;

    std::vector<int> structure = {1,1,1};
    PyNet net = PyNet(structure);
    for(int i = 0; i < NUM_TESTS; i++){
       ins.emplace_back(1,1);
       actuals.emplace_back(1,1);
       actuals.at(i).setRawElement(0,i*2);
       ins.at(i).setRawElement(0,i);
    }
    net.randomiseParams();
    for(int epoch = 0; epoch < 500; epoch++) {
        for (int i = 0; i < NUM_TESTS; i++) {
            Matrix out = net.feedFoward(ins.at(i));
            Matrix loss = mseLossDerivitive(out, actuals.at(i));
            Matrix actualLoss = mseLoss(out,actuals.at(i));
            std::cout << "\nout : ";
            out.printMatrix();
            std::cout << "loss : ";
            actualLoss.printMatrix();
            std::cout << "i : " << i << "\n";
            net.updateGradients(loss, ins.at(i));
            net.applyGradients();
            net.clearGradients();
        }
    }
    std::cout << "\ntest : ";
    net.feedFoward(ins.at(2)).printMatrix();
#endif
}