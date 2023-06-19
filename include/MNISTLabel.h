//
// Created by alan on 19/06/23.
//

#ifndef CUSTOMNN_MNISTLABEL_H
#define CUSTOMNN_MNISTLABEL_H

#include "matrix.h"
struct  MNISTLabel{
    MNISTLabel(int rowsIn,int colsIn, int rowsOut ,int colsOut);
    Matrix _in{};
    Matrix _out{};
};

std::vector<MNISTLabel> parseMNISTLabels(std::string filePath);
#endif //CUSTOMNN_MNISTLABEL_H
