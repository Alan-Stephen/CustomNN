#include "../include/matrix.h"

Matrix::Matrix(int rows, int columns)
{
	data = std::make_unique<double[]>(rows * columns);
	this->rows = rows; // rows
	this->columns = columns; // columns
	for (int i = 0; i < rows * columns; i++){
		setRawElement(i, 0);
	}
}

void Matrix::printMatrix() const
{
    for (int i = 0; i < rows * columns; i++)
	{
		if (i % columns == 0)
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

void Matrix::setRawElement(int i, double value)
{
	data[i] = value;
}

int Matrix::coordsToRaw(int x, int y, int coll)
{
	return (((y - 1) * coll) + x) - 1;
}

double Matrix::getElement(int x, int y) const
{
	return getRawElement(y * this->columns + x);
}

inline void Matrix::setElement(int x, int y, double value)
{
	setRawElement(y * this->columns + x, value);
}

// todo this->data may be already initliased.
Matrix::Matrix(const Matrix& other){
	std::cout << "\nmaking copy";
	this->rows = other.rows;
	this->columns = other.columns;
	this->data = std::make_unique<double[]>(other.rows * other.columns);
	for(int i = 0; i < other.rows * other.columns; i++){
		this->data[i] = other.data[i];
	}	
}

Matrix &Matrix::operator=(const Matrix &matrix) {
    this->rows = matrix.rows;
    this->columns = matrix.columns;

    this->data = std::make_unique<double[]>(this->rows*this->columns);
    for(int i = 0; i < rows * columns; i++){
        this->setRawElement(i,matrix.getRawElement(i));
    }
}

void copyMatrix(const Matrix &from, Matrix &to) {
    to.rows = from.rows;
    to.columns = from.columns;

    to.data = std::make_unique<double[]>(from.rows * from.columns);
    for(int i = 0; i < from.columns * from.rows;i++) {
        to.setRawElement(i,from.getRawElement(i));
    }
}

void addMatrix(const Matrix &a, const Matrix &b, Matrix &out) {
    if (a.rows != b.rows || a.columns != b.columns) {
        std::cout << "[ERROR] cannot add matrix invalid shapes";
        exit(1);
    }

    for(int i = 0; i < a.rows * a.columns; i++){
        double value = a.getRawElement(i) + b.getRawElement(i);
        out.setRawElement(i,value);
    }
}

 Matrix multiplyMatrix(const Matrix &a, const Matrix &b){
    if(a.columns != b.rows){
        std::cout << "ERROR: invalid size of matrix for multiplication\n";
        std::exit(1);
    }

    Matrix out = Matrix(a.rows,b.columns);
    for(int row = 0; row < a.rows; row++){
        for(int col = 0; col < b.columns; col++){
            double sum = 0;
            for(int k = 0; k < a.columns; k++){
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
    for(int i = 0; i < a.columns * a.rows; i++){
        a.setRawElement(i,((double) rand() / RAND_MAX));
    }
}

/**
 * Matrix minus in the form a - b;
 *
 * */
Matrix minusMatrix(const Matrix &a, const Matrix &b){
    if(a.rows != b.rows || a.columns != b.columns){
        std::cout << "invalid shapes for minus matrix\n";
        exit(1);
    }

    Matrix result = Matrix(a.rows,a.columns);
    for(int i = 0; i < a.rows * a.columns; i++){
        result.setRawElement(i,a.getRawElement(i) - b.getRawElement(i));
    }
    return result;
}