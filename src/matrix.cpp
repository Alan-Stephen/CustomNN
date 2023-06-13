#include <cmath>
#include "../include/matrix.h"

//todo : make raw elements of matrix iterable, replace double list with std::array
Matrix::Matrix(int rows, int columns)
{
	data = std::make_unique<double[]>(rows * columns);
	this->numRows = rows; // numRows
	this->numColumns = columns; // numColumns
	for (int i = 0; i < rows * columns; i++){
		setRawElement(i, 0);
	}
}

void Matrix::printMatrix() const
{
    for (int i = 0; i < numRows * numColumns; i++)
	{
		if (i % numColumns == 0)
		{
			std::cout << '\n';
		}
		std::cout << this->getRawElement(i) << " ";
	}
	std::cout << "\n";
}

double Matrix::getRawElement(int i) const
{
	return data[i];
}

void Matrix::setRawElement(const int i,const double value)
{
	data[i] = value;
}

int Matrix::coordsToRaw(const int x,const int y,const int coll)
{
	return (((y - 1) * coll) + x) - 1;
}

double Matrix::getElement(int x, int y) const
{
	return getRawElement(y * this->numColumns + x);
}

inline void Matrix::setElement(int x, int y, double value)
{
	setRawElement(y * this->numColumns + x, value);
}

Matrix::Matrix(const Matrix& other){
	std::cout << "\nmaking copy";
	this->numRows = other.numRows;
	this->numColumns = other.numColumns;
	this->data = std::make_unique<double[]>(other.numRows * other.numColumns);
	for(int i = 0; i < other.numRows * other.numColumns; i++){
		this->data[i] = other.data[i];
	}	
}

Matrix &Matrix::operator=(const Matrix &matrix) {
    this->numRows = matrix.numRows;
    this->numColumns = matrix.numColumns;

    this->data = std::make_unique<double[]>(this->numRows * this->numColumns);
    for(int i = 0; i < numRows * numColumns; i++){
        this->setRawElement(i,matrix.getRawElement(i));
    }
}

void Matrix::add(const Matrix &a) {
    if(not isSameDimensions(*this,a)){
        std::cout << "[ERROR] wrong matrix dimensions for inline add Matrixes\n";
        exit(1);
    }

    for(int i = 0; i < a.numColumns * a.numRows;i++){
        double value = a.getRawElement(i) + this->getRawElement(i);
        this->setRawElement(i,value);
    }
}
void Matrix::minus(const Matrix &a) {
    if(not isSameDimensions(*this,a)){
        std::cout << "[ERROR] wrong matrix dimensions for inline add Matrixes\n";
        exit(1);
    }

    for(int i = 0; i < a.numColumns * a.numRows;i++){
        double value = a.getRawElement(i) - this->getRawElement(i);
        this->setRawElement(i,value);
    }
}

void copyMatrix(const Matrix &from, Matrix &to) {
    to.numRows = from.numRows;
    to.numColumns = from.numColumns;

    to.data = std::make_unique<double[]>(from.numRows * from.numColumns);
    for(int i = 0; i < from.numColumns * from.numRows; i++) {
        to.setRawElement(i,from.getRawElement(i));
    }
}

void addMatrix(const Matrix &a, const Matrix &b, Matrix &out) {
    if (a.numRows != b.numRows || a.numColumns != b.numColumns) {
        std::cout << "[ERROR] cannot add matrix invalid shapes";
        exit(1);
    }

    for(int i = 0; i < a.numRows * a.numColumns; i++){
        double value = a.getRawElement(i) + b.getRawElement(i);
        out.setRawElement(i,value);
    }
}

 Matrix multiplyMatrix(const Matrix &a, const Matrix &b){
    if(a.numColumns != b.numRows){
        std::cout << "ERROR: invalid size of matrix for multiplication a :" << a.numRows << " " << a.numColumns <<
            " b: " << b.numRows << " " << b.numColumns << "\n";
        std::exit(1);
    }

    Matrix out = Matrix(a.numRows, b.numColumns);
    for(int row = 0; row < a.numRows; row++){
        for(int col = 0; col < b.numColumns; col++){
            double sum = 0;
            for(int k = 0; k < a.numColumns; k++){
                sum += a.getElement(k,row) * b.getElement(col,k);
            }
            out.setElement(col,row,sum);
        }
    }

    return out;
}

/**
 * Sets values in matrix to a random number bewteen 0 < x < 1
 * Matrix must be initlaised otherwise seg fault will be thrown.
 * */
void randomizeMatrix(Matrix &a){
    for(int i = 0; i < a.numColumns * a.numRows; i++){
        a.setRawElement(i,((double) rand() / RAND_MAX));
    }
}

/**
 * Matrix minus in the form a - b;
 *
 * */
Matrix minusMatrix(const Matrix &a, const Matrix &b){
    if(a.numRows != b.numRows || a.numColumns != b.numColumns){
        std::cout << "invalid shapes for minus matrix\n";
        exit(1);
    }

    Matrix result = Matrix(a.numRows, a.numColumns);
    for(int i = 0; i < a.numRows * a.numColumns; i++){
        result.setRawElement(i,a.getRawElement(i) - b.getRawElement(i));
    }
    return result;
}

Matrix multiplyMatrix(const double scalar, const Matrix &a) {
   Matrix matrix = Matrix(a.numRows, a.numColumns);
   for(int i = 0; i < a.numRows * a.numColumns; i++){
       matrix.setRawElement(i,a.getRawElement(i) * scalar);
   }
   return matrix;
}

Matrix addMatrix(const Matrix &a, const Matrix &b){
    if(a.numColumns != b.numColumns || a.numRows != b.numRows){
        std::cout << "INVALID SIZES FOR addMatrix()";
        exit(1);
    }

    Matrix output = Matrix(a.numRows,a.numColumns);
    for(int i = 0; i < a.numColumns * a.numRows;i++){
        double value = a.getRawElement(i) + b.getRawElement(i);
        output.setRawElement(i,value);
    }
    return output;
}

Matrix hadamardProduct(const Matrix &a, const Matrix &b){
    if(a.numColumns != b.numColumns || a.numRows != b.numRows){
        std::cout << "INVALID SIZES FOR hadamardProduct()";
        exit(1);
    }

    Matrix output = Matrix(a.numRows,a.numColumns);
    for(int i = 0; i < a.numColumns * a.numRows;i++){
        double value = a.getRawElement(i) * b.getRawElement(i);
        output.setRawElement(i,value);
    }
    return output;
}

Matrix mseLoss(const Matrix &pred, const Matrix &actual){
    if(pred.numColumns != actual.numColumns || pred.numRows != actual.numRows){
        std::cout << "INVALID SIZES FOR mseLoss()";
        exit(1);
    }

    Matrix temp = Matrix(pred.numRows,pred.numColumns);
    for(int i = 0; i < pred.numColumns * pred.numRows; i++){

        double value = pow(pred.getRawElement(i) - actual.getRawElement(i),2.0F);
        temp.setRawElement(i,value);
    }
    return temp;
}

bool isSameDimensions(const Matrix &a, const Matrix &b) {
    if(a.numColumns != b.numColumns || a.numRows != b.numRows){
        return false;
    }

    return true;
}

Matrix mseLossDerivitive(const Matrix &pred, const Matrix &actual) {
    return multiplyMatrix(2, minusMatrix(actual,pred));
}
