#include <iostream>
#include <vector>
#include <algorithm>
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

    const double learningRate = 1;
    const double leakValue = 0.1;
    const int batchSize = 16;

    PyNet net = PyNet();

    net.addLayer(new LinearLayer(784,100,learningRate,batchSize));
    net.addLayer(new SigmoidLayer(100));
    net.addLayer(new LinearLayer(100,10,learningRate,batchSize));
    net.addLayer(new SoftmaxLayer(10));

    // todo: randomise data as it's currently goes 9 8 7 6 5 4 3 2 1 : maybe see if it will improve peformance
    std::vector<MNISTLabel> data = parseMNISTLabels("../data/mnist_train.csv");
    net.randomiseParams();

    int batchCorrect = 0;
    int printCounter = 0;
    Matrix lossTotal(data.at(0)._out.numRows,1);

    const int NUM_REPS = 3;
    const int PRINT_FREQ = 8;
    for(int reps = 0; reps < NUM_REPS; reps++){
        int batch = 0;
        for(MNISTLabel label: data){
            batch++;

            Matrix &actual = label._out;
            Matrix pred = net.feedFoward(label._in);

            if(pred.maxIndex() == actual.maxIndex())
                batchCorrect++;

            Matrix error = crossEntropyLossDeriv(pred,actual);
            lossTotal.add(crossEntropyLoss(pred,actual));

            net.updateGradients(error,label._in);

            if(batch % batchSize == 0){
                net.applyGradients();
                net.clearGradients();
                batch = 0;
                printCounter++;


                if(printCounter % PRINT_FREQ == 0) {
                    printCounter = 0;
                    std::cout << "REP : " << reps << ", loss : " << sum(lossTotal) / (batchSize * 10 * PRINT_FREQ)
                              << ", batch correctness : " << (((double) batchCorrect) / (((double) batchSize) * PRINT_FREQ)) * 100
                              << "\n";

                    batchCorrect = 0;
                    clear(lossTotal);
                }
            }
        }
    }

    std::vector<MNISTLabel> testing_data = parseMNISTLabels("../data/mnist_test.csv");

    int correct = 0;
    for(MNISTLabel label: testing_data){
        Matrix out = net.feedFoward(label._in);
        if(out.maxIndex() == label._out.maxIndex())
            correct++;
    }

    std::cout << "\n CORRECTNESS ON TEST : " <<  ((double) correct / (double) testing_data.size()) * 100 << "\n";
}