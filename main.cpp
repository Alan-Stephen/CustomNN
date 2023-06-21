#include <iostream>
#include <vector>
#include "include/matrix.h"
#include "include/nueral_network.h"
#include "include/LinearLayer.h"
#include "include/py_net.h"
#include "include/MNISTLabel.h"
#include "include/SigmoidLayer.h"

// todo: do a lot of refactoring
double functionToEstimate(const double in){
    return in * 2;
}


int main(int argc, char const *argv[]){
    const double learningRate = 1;
    // Building network
    PyNet net = PyNet();
    net.addLayer(new SigmoidLayer(784));
    net.addLayer(new LinearLayer(784,10,learningRate));
    net.addLayer(new SigmoidLayer(10));


    // todo: randomise data as it's currently goes 9 8 7 6 5 4 3 2 1 : maybe see if it will improve peformance
    std::vector<MNISTLabel> data = parseMNISTLabels("../data/mnist_train.csv");



    net.randomiseParams();

    int right = 0;
    int total = 0;

    for (int i = 0; i < data.size(); i++) {
        Matrix out = net.feedFoward(data.at(i)._in);

        if(data.at(i)._out.maxIndex() == out.maxIndex()){
           right++;
        }

        Matrix loss = mseLossDerivitive(out, data.at(i)._out);
        if (i % 1000 == 0) {
            out.printMatrix();
            data.at(i)._out.printMatrix();
            Matrix actualLoss = mseLoss(out, data.at(i)._out);
            double averageLoss = sum(actualLoss) / 10;
            std::cout << right << " " << total << "\n";
            std::cout << "Percentage right: " << ((double) right/(double) i) * 100.0 << "\n";
            std::cout << "loss : " << averageLoss << "\n";
            std::cout << "i : " << i << "\n";
        }
        net.updateGradients(loss, data.at(i)._in);
        net.applyGradients();
        net.clearGradients();
    }
#if 0
    Matrix test = Matrix(10,10);
    randomizeMatrix(test);

    std::cout << test.data.size();

    const int NUM_TESTS = 10;
    std::vector<Matrix> ins;
    std::vector<Matrix> actuals;

    for(int i = 0; i < NUM_TESTS; i++){
       ins.emplace_back(1,1);
       actuals.emplace_back(1,1);
       actuals.at(i).setRawElement(0,i*2);
       ins.at(i).setRawElement(0,i);
    }
    net.randomiseParams();
    for(int epoch = 0; epoch < 5000; epoch++) {
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