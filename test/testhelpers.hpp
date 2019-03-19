#ifndef TESTHELPERS_HPP
#define TESTHELPERS_HPP
#include <iostream>
#include <cstdlib>
#include <cassert>

namespace gemmbitserial {

// Generate a random vector of -1 and +1 values of given dimension
template <typename T>
void generateRandomVector_Bipolar(size_t dim, T * ret) {
  for(size_t i = 0; i < dim; i++) {
    ret[i] = (rand() % 2 == 0) ? 1 : -1;
  }
}

/**
* Generate a random vector with given dimension and number of bits
*/
template <typename T>
void generateRandomVector(size_t bits, size_t dim, T * ret, bool allowNeg = false) {
  assert(bits <= (sizeof(T) * 8));
  if(bits == 1 && allowNeg) {
    // generate bipolar values
    generateRandomVector_Bipolar(dim, ret);
    return;
  }
  int32_t minVal = 0;
  int32_t maxVal = (1 << bits);
  for(size_t i = 0; i < dim; i++) {
    ret[i] = (rand() % maxVal) - (allowNeg ? maxVal/2 : 0);
  }
}

template <typename LHSType, typename RHSType>
void naive_int_gemm(LHSType * lhs, RHSType * rhs, int32_t * res, int rows, int depth, int cols) {
  for(int k = 0; k < cols; k++) {
    for(int i = 0; i < rows; i++) {
      int32_t acc = 0;
      for(int j = 0; j < depth; j++) {
        acc += lhs[i * depth + j] * rhs[k * depth + j];
      }
      res[k * rows + i] = acc;
    }
  }
}

template <typename T>
void naive_sum_rows(T * m, int32_t * res, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    int32_t acc = 0;
    for(int k = 0; k < cols; k++) {
      acc += m[i * cols + k];
    }
    res[i] = acc;
  }
}

template <typename T>
void printmatrix(T * mat, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      std::cout << (int) mat[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void printmatrixdiff(const T * mat1, const T * mat2, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      if(mat1[i * cols + j] != mat2[i * cols + j]) {
        std::cout << "Difference at (i,j) = " << i << " " << j << " Mat1: " << (int)mat1[i * cols + j] << " Mat2: " << mat2[i * cols + j] << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

}
#endif
