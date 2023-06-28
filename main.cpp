#include <iostream>
#include <vector>
#include "include/matrix.h"
#include "include/nueral_network.h"
#include "include/LinearLayer.h"
#include "include/py_net.h"
#include "include/MNISTLabel.h"
#include "include/SigmoidLayer.h"
#include "include/ReLuLayer.h"
#include "include/SoftmaxLayer.h"

double functionToEstimate(const double in){
    return in * 2;
}



int main(int argc, char const *argv[]){
    // with minibatching you delay application of gradients, apparently this is faster.
    const double learningRate = 1;
    double leakValue = 0.1;
    const int batchSize = 32;
    // Building network
    PyNet net = PyNet();
    net.addLayer(new LinearLayer(784,10,learningRate,batchSize));
    net.addLayer(new ReLuLayer(10,leakValue));
    net.addLayer(new LinearLayer(10,10,learningRate,batchSize));
    net.addLayer(new SoftmaxLayer(10));

    // todo: randomise data as it's currently goes 9 8 7 6 5 4 3 2 1 : maybe see if it will improve peformance
    std::vector<MNISTLabel> data = parseMNISTLabels("../data/mnist_train.csv");



    net.randomiseParams();

    int right = 0;
    int batchCorrect = 0;
    Matrix lossTotal(data.at(0)._out.numRows,1);

    for (int i = 0; i < data.size(); i++) {
        Matrix &inRef = data.at(i)._in;

        Matrix out = net.feedFoward(inRef);

        if(data.at(i)._out.maxIndex() == out.maxIndex()){
            batchCorrect++;
            right++;
        }

        Matrix loss = mseLossDerivitive(out,data.at(i)._out);
        Matrix actualLoss = mseLoss(data.at(i)._out,out);
        lossTotal.add(actualLoss);
        net.updateGradients(loss,inRef);
        if (i % batchSize == 0) {
            net.applyGradients();
            net.clearGradients();
            out.printMatrix();
            std::cout << "-------------------------" << "\n";
            data.at(i)._out.printMatrix();
            double averageLoss = sum(lossTotal) / (10.0 * batchSize);
            clear(lossTotal);
            std::cout << right << " " << i << "\n";
            std::cout << "Batch correctness : " << ((double) batchCorrect / (double) batchSize) * 100.0 << "\n";
            std::cout << "Percentage right: " << ((double) right/(double) i) * 100.0 << "\n";
            std::cout << "loss : " << averageLoss << "\n";
            std::cout << "i : " << i << "\n";
            batchCorrect = 0;
        }
    }

    right = 0;
    std::vector<MNISTLabel> data2 = parseMNISTLabels("../data/mnist_test.csv");
    for (int i = 0; i < data2.size(); ++i) {
        Matrix &inRef = data2.at(i)._in;
        Matrix out = net.feedFoward(inRef);

        if(data2.at(i)._out.maxIndex() == out.maxIndex()){
            right++;
        }
    }

    std::cout << "TOTAL RIGHT: " << ((double) right) / ((double) data2.size()) * 100 << "\n";
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