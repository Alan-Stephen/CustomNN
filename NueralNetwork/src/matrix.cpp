#include "matrix.h"

Matrix::Matrix(int n, int m)
{
	data = std::make_unique<double[]>(n * m);
	this->n = n; // rows
	this->m = m; // columns
	for (int i = 0; i < n * m; i++)
	{
		setRawElement(i, 0);
	}
}

void Matrix::printMatrix()
{
	int i;
	std::cout << "printing matrix\n"; 	
	for (int i = 0; i < n * m; i++)
	{
		if (i % m == 0)
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

Matrix Matrix::Multiply(double scalar) const
{
	Matrix matrix(this->n, this->m);
	int i;
	for (i = 0; i < matrix.n * matrix.m; i++)
	{
		matrix.setRawElement(i, this->getRawElement(i) * scalar);
	}
	return matrix;
}

int Matrix::coordsToRaw(int x, int y, int rowl, int coll) const
{
	return (((y - 1) * coll) + x) - 1;
}

double Matrix::getElement(int x, int y) const
{
	return getRawElement(y * this->m + x);
}

inline void Matrix::setElement(int x, int y, double value)
{
	setRawElement(y * this->m + x, value);
}

Matrix Matrix::operator*(const Matrix &other) const {
	return this->Multiply(other);	
}


Matrix Matrix::operator+(const Matrix &other) const{
	int i;
	Matrix matrix(other.n,other.m);
	if(other.n != this->n || other.m != this->m) {
		std::cout << "[ERROR] cannot add matrix invalid shapes";
		return matrix;
	}

	for(i = 0; i < other.n * other.m; i++) {
		matrix.setRawElement(i, this->getRawElement(i) + other.getRawElement(i));
	}
	return matrix;
} 

Matrix Matrix::operator*(const double scalar) const{
	return this->Multiply(scalar);
}

Matrix Matrix::Multiply(const Matrix &other) const{
	Matrix matrix(this->n, other.m);
	int row, col, k;
	double sum = 0;
	if ((other.n != this->m))
	{
		std::cout << "[ERROR] invalid sizes";
		return matrix;
	}

	for (row = 0; row < this->n; row++)
	{
		for (col = 0; col < other.m; col++)
		{
			sum = 0;
			for (k = 0; k < this->m; k++)
			{
				sum += this->getElement(k, row) * other.getElement(col, k);
			}
			matrix.setElement(col, row, sum);
		}
	}
	return matrix;
}
// Random Matrix

RandomMatrix::RandomMatrix(int n,int m){
	data = std::make_unique<double[]>(n * m);
	this->n = n; // rows
	this->m = m; // columns
	for (int i = 0; i < n * m; i++)
	{
		setRawElement(i, double (rand()) / RAND_MAX);
	}
}





	