#ifndef MATRICES_H
#define MATRICES_H

// implement matrix structure

typedef struct Matrix {
  unsigned int rows;
  unsigned int columns;
  double* matrix_data;
} Matrix;

double drand();

double random_normal();

// implement matrix calculations

Matrix* create_empty_matrix(unsigned int rows, unsigned int columns);

void free_matrix(Matrix* matrix);

double get_element(Matrix* mat, unsigned int row, unsigned int column);

Matrix* get_column(Matrix* mat, unsigned int column);

Matrix* get_row(Matrix* mat, unsigned int row);

double sum(Matrix* mat);

void hadamard_product(Matrix* mat1, Matrix* mat2, Matrix* matAns);

double dot_product(Matrix* mat1, Matrix* mat2);

void print_matrix(Matrix* mat);

Matrix* multiply(Matrix* mat1, Matrix* mat2);

void transpose(Matrix* mat, Matrix* matAns);

void outer_product(Matrix* mat1, Matrix* mat2, Matrix* matAns);

void add(Matrix* mat1, Matrix* mat2, Matrix* matAns);

void copy_matrix(Matrix* mat, Matrix* matAns);

#endif