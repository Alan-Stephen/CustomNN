#include <iostream>
#include <vector>
#include "include/matrix.h"
#include "include/nueral_network.h"

// todo do a lot of refactoring
double functionToEstimate(const double in){
    return in * 2;
}


int main(int argc, char const *argv[]){
    std::vector<int> structure = std::vector<int>({1});
    NueralNetwork network = NueralNetwork(1,structure,1);

    network.randomizeWeightMatrix();

    Matrix input = Matrix(1,1);
    randomizeMatrix(input);

    Matrix output = network.feedFoward(input);
    Matrix actualOutput = Matrix(1,1);
    actualOutput.setRawElement(0,input.getRawElement(0)*3);
    std::cout << "\ninput ";
    input.printMatrix();
    std::cout << "output ";
    output.printMatrix();

    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);
    network.updateWeights(input,actualOutput);

    Matrix output2 = network.feedFoward(input);
    std::cout << " output2";
    output2.printMatrix();

}