//
// Created by alan on 20/06/23.
//

#include <vector>
#include <fstream>
#include "../include/MNISTLabel.h"


std::vector<MNISTLabel> parseMNISTLabels(std::string filePath) {
    std::vector<MNISTLabel> parseMNISTLabels(std::string filePath);
    std::vector<MNISTLabel> output;
    std::ifstream inputFile(filePath);

    std::string line;
    std::getline(inputFile,line);
    while (std::getline(inputFile, line)){
        // one hot encoded output values
        output.emplace_back(784,1,10,1);
        int endPos = -1;
        int startPos = 0;

        endPos = line.find(',');
        std::string value = line.substr(startPos,endPos-startPos);
        startPos = endPos;
        int parsedValue = std::stoi(value);
        output.back()._out.setRawElement(parsedValue,parsedValue);
        const int NUM_FEATURES = 724;
        for(int i = 0; i < NUM_FEATURES; i++){
            startPos++;
            endPos = line.find(',',startPos);

            value = line.substr(startPos,endPos-startPos);
            output.back()._in.setRawElement(i,std::stoi(value));
            startPos = endPos;
        }
    }
    inputFile.close();
    return output;
}

MNISTLabel::MNISTLabel(int rowsIn, int colsIn, int rowsOut, int colsOut): _in(rowsIn,colsIn), _out(rowsOut,colsOut) {}
