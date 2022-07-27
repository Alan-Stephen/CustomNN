#pragma once
#include <iostream>
#include <memory>
#include <chrono>
//TODO CHAGE TO CONST REFERENCES OT PREVENT COPYING

class Matrix {
public:
	int n;
	int m;

	Matrix(int x, int y);

	Matrix operator+(const Matrix &other) const; 
	Matrix Add(const Matrix &other) const;
	Matrix Multiply(const double scalar) const; 
	Matrix Multiply(const Matrix &other) const; 
	inline double getElement(int x,int y) const;
	void printMatrix();
	inline void setElement(int x, int y,double value);
	Matrix operator*(const double scalar) const;
	Matrix operator*(const Matrix &other ) const;
protected:
	Matrix(){}
	inline double getRawElement(int i) const;
	inline void setRawElement(int i,double value);
	std::unique_ptr<double[]> data;
	inline int coordsToRaw(int x, int y, int rowl,int coll) const;

};

class RandomMatrix : public Matrix {
public:
	RandomMatrix(int n,int m); 
};