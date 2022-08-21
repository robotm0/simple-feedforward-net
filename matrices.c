#include "matrices.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// generate normal distribution for starting values

double drand() {
  // uniform distribution, (0..1]
  double result = (rand()+1.0)/(RAND_MAX+1.0);
  return result;
}

double random_normal() {
  // normal distribution, centered on 0, std dev 1
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

// implement matrix calculations

Matrix* create_empty_matrix(unsigned int rows, unsigned int columns) {
  Matrix* matAns = (Matrix*)malloc(sizeof(Matrix));
  matAns->rows = rows;
  matAns->columns = columns;
  unsigned int matrix_size = matAns->rows * matAns->columns;
  matAns->matrix_data = (double*)calloc(sizeof(double), matrix_size);
  return matAns;
}

void free_matrix(Matrix* matrix) {
  free(matrix->matrix_data);
  free(matrix);
}

double get_element(Matrix* mat, unsigned int row, unsigned int column) {
  unsigned int index = (mat->columns * row) + column;
  return mat->matrix_data[index];
}

Matrix* get_column(Matrix* mat, unsigned int column) {
  Matrix* matAns = (Matrix*)malloc(sizeof(Matrix) * mat->rows);
  matAns->rows = mat->rows;
  matAns->columns = 1;
  double* columnData = (double*)malloc(sizeof(double) * matAns->rows);
  for (unsigned int r = 0; r < matAns->rows; r++) {
    columnData[r] = get_element(mat, r, column);
  }
  matAns->matrix_data = columnData;
  return matAns;
}

Matrix* get_row(Matrix* mat, unsigned int row) {
  Matrix* matAns = (Matrix*)malloc(sizeof(Matrix) * mat->columns);
  matAns->rows = 1;
  matAns->columns = mat->columns;
  double* rowData = (double*)malloc(sizeof(double) * matAns->columns);
  for (unsigned int c = 0; c < matAns->columns; c++) {
    rowData[c] = get_element(mat, row, c);
  }
  matAns->matrix_data = rowData;
  return matAns;
}

double sum(Matrix* mat) {
  // gets sum of elements in matrix
  float total = 0;
  unsigned int mat_size = mat->rows * mat->columns;
  for (unsigned int i = 0; i < mat_size; i++) {
    total += mat->matrix_data[i];
  }
  return total;
}

void hadamard_product(Matrix* mat1, Matrix* mat2, Matrix* matAns) {
  if (mat1->rows != mat2->rows || mat1->columns != mat2->columns) {
    printf("Error: Hadamard product: "
    "matrices not the same size: %d x %d, %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns
    );
    exit(1);
  } else {
    matAns->rows = mat1->rows;
    matAns->columns = mat1->columns;
    unsigned int matrix_size = mat1->rows * mat1->columns;
    for (unsigned int i = 0; i < matrix_size; i++) {
      matAns->matrix_data[i] = mat1->matrix_data[i] * mat2->matrix_data[i];
    }
  }
}

double dot_product(Matrix* mat1, Matrix* mat2) {
  unsigned int mat1_size = mat1->columns * mat1->rows;
  unsigned int mat2_size = mat2->columns * mat2->rows;
  if (mat1_size != mat2_size) {
    printf("Error: Dot product: "
    "matrices not the same size: %d x %d, %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns);
    exit(1);
  } else {
    double dot_product = 0;
    for (unsigned int i = 0; i < mat1_size; i++) {
      dot_product += (mat1->matrix_data[i] * mat2->matrix_data[i]);
    }
    return dot_product;
  }
}

void print_matrix(Matrix* mat) {
  printf("Matrix dimensions: %d x %d\n", mat->rows, mat->columns);
  for (unsigned int r = 0; r < mat->rows; r++) {
    printf("| ");
    for (unsigned int c = 0; c < mat->columns; c++) {
      printf("%f ", get_element(mat, r, c));
    }
    printf("|\n");
  }
}

Matrix* multiply(Matrix* mat1, Matrix* mat2) {
  // to multiply a matrix, mat1->columns == mat2->rows
  if (mat1->columns != mat2->rows) {
    printf("Error: Matrix multiplication: "
    "matrices incompatible: %d x %d, %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns);
    exit(1);
  } else {
    // new matrix dimensions: mat1.rows, mat2.columns
    // matAns->rows = mat1->rows;
    // matAns->columns = mat2->columns;
    // unsigned int matrix_size = matAns->rows * matAns->columns;
    // free(matAns->matrix_data);
    // free(matAns);
    Matrix* mat = create_empty_matrix(
      mat1->rows,
      mat2->columns
    );
    for (unsigned int r = 0; r < mat->rows; r++) {
      // r^th row of first matrix, c^th column of second matrix
      Matrix* row = get_row(mat1, r);
      for (unsigned int c = 0; c < mat->columns; c++) {
        Matrix* col = get_column(mat2, c);
        float dot = 0;
        dot = dot_product(row, col);
        unsigned int index = (r * mat->columns) + c;
        mat->matrix_data[index] = dot;
        free_matrix(row);
        free_matrix(col);
      }
    }
    return mat;
  }
}

void transpose(Matrix* mat, Matrix* matAns) {
  unsigned int rows = mat->columns;
  unsigned int columns = mat->rows;
  matAns->rows = rows;
  matAns->columns = columns;
  // unsigned int matrix_size = matAns->rows * matAns->columns;
  for (unsigned int r = 0; r < mat->rows; r++) {
    for (unsigned int c = 0; c < mat->columns; c++) {
      // usually it would be (columns * r) + c
      unsigned int transpose_index = (columns * c) + r;
      matAns->matrix_data[transpose_index] = get_element(mat, r, c);
    }
  }
}

void outer_product(Matrix* mat1, Matrix* mat2, Matrix* matAns) {
  if (mat1->rows != matAns->rows || mat2->rows != matAns->columns) {
    printf("Error: Outer product: "
    "matrices not the right size: %d x %d, %d x %d, MatAns: %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns,
    matAns->rows, matAns->columns
    );
    exit(1);
  } else if (mat1->columns != 1 || mat2->columns != 1) {
    printf("Error: Outer product: "
    "matrices not vectors: %d x %d, %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns
    );
    exit(1);
  } else {
    unsigned int mat1_size = mat1->rows;
    unsigned int mat2_size = mat2->rows;
    matAns->rows = mat1_size;
    matAns->columns = mat2_size;
    // unsigned int matrix_size = matAns->rows * matAns->columns;
    for (unsigned int i = 0; i < mat1_size; i++) {
      for (unsigned int j = 0; j < mat2_size; j++) {
        unsigned int index = (matAns->columns * i) + j;
        matAns->matrix_data[index] = (
          mat1->matrix_data[i] * mat2->matrix_data[j]
        );
      }
    }
  }
}

void add(Matrix* mat1, Matrix* mat2, Matrix* matAns) {
  if (mat1->rows != mat2->rows || mat1->columns != mat2->columns) {
    printf("Error: Matrix addition: "
    "matrices not the same size: %d x %d, %d x %d\n",
    mat1->rows, mat1->columns, mat2->rows, mat2->columns
    );
    exit(1);
  } else {
    matAns->rows = mat1->rows;
    matAns->columns = mat1->columns;
    unsigned int matrix_size = mat1->rows * mat1->columns;
    // free(matAns->matrix_data);
    // double* matAns_data = (double*)malloc(sizeof(double) * matrix_size);
    for (unsigned int i = 0; i < matrix_size; i++) {
      matAns->matrix_data[i] = mat1->matrix_data[i] + mat2->matrix_data[i];
    }
    // matAns->matrix_data = matAns_data;
  }
}

void copy_matrix(Matrix* mat, Matrix* matAns) {
  if (mat->rows != matAns->rows || mat->columns != matAns->columns) {
    printf("Error: Matrix copy: "
    "matrices not the same size: %d x %d, %d x %d\n",
    mat->rows, mat->columns, matAns->rows, matAns->columns
    );
    exit(1);
  } else {
    unsigned int matrix_size = mat->rows * mat->columns;
    // free(matAns->matrix_data);
    // double* matAns_data = (double*)malloc(sizeof(double) * matrix_size);
    for (unsigned int i = 0; i < matrix_size; i++) {
      matAns->matrix_data[i] = mat->matrix_data[i];
    }
    // matAns->matrix_data = matAns_data;
  }
}