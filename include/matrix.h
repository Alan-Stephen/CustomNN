#pragma once
#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
//TODO CHAGE TO CONST REFERENCES OT PREVENT COPYING
// TODO MAKE A LOT OF THESE CLASSES STATIC, MAKE THEM USE AN OUT VARIALBE INSTEAD
// TODO: MAKE NAMESPACE MATRIX FOR MATRIX FUNCTOINS, make this more imperitive.
// TODO: Iterators?
#ifndef CUSTOMNN_MATRIX_H
#define CUSTOMNN_MATRIX_H

class Matrix {
public:
    bool isTransposed = false;
    std::vector<double> data;
    int numRows;
	int numCols;

    size_t size() const;

    explicit Matrix(int rows = 0,int columns = 0);
    void add(const Matrix &a);
	[[nodiscard]] inline double getElement(int cols, int rows) const;
    void printMatrix() const;
    inline void setElement(int cols, int rows, double value);
    inline void setRawElement(int i,double value);
    [[nodiscard]] inline double getRawElement(int i) const;
    Matrix(const Matrix& other);
    Matrix &operator=(const Matrix &matrix);
    void minus(const Matrix &a);
    void transpose();

protected:
    static inline int coordsToRaw(int x, int y,int coll) ;

};

Matrix hadamardProduct(const Matrix &a, const Matrix &b);
Matrix minusMatrix(const Matrix &a, const Matrix &b);
void copyMatrix(const Matrix &from, Matrix &to);
Matrix multiplyMatrix(const Matrix &a, const Matrix &b);
Matrix multiplyMatrix(const double scalar, const Matrix &a);
Matrix addMatrix(const Matrix &a, const Matrix &b);
void randomizeMatrix(Matrix &a);
Matrix mseLoss(const Matrix &pred, const Matrix &actual);
bool isSameDimensions(const Matrix &a, const Matrix &b);
Matrix mseLossDerivitive(const Matrix &pred, const Matrix &actual);
Matrix dot(const Matrix &a,const Matrix &b);
double sum(const Matrix &a);

#endif