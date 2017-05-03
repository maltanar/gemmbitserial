#include <cassert>
#include "convert.h"

/**
* Convert a buffer of unsigned char values into a gemm-bitserial vector
*/
BitSerialVector toBitSerialVector(const uint8_t * vec, const size_t n, const size_t bits) {
  BitSerialVector ret;
  BitVector currentBitGroup(n);

  for(size_t b = 0; b < bits; b++) {
    currentBitGroup.clear();
    uint8_t currentMask = 1 << b;
    for(size_t i = 0; i < n; i++) {
      if((vec[i] & currentMask) != 0) {
        currentBitGroup.add(i);
      }
    }
    ret.push_back(currentBitGroup);
  }
  return ret;
}

/**
* Convert a gemm-bitserial vector into a buffer of unsigned char values
*/
void fromBitSerialVector(const BitSerialVector & vec, uint8_t * ret) {
  const size_t precision = vec.size();
  const size_t n = vec[0].size();
  for(size_t i = 0; i < n; i++) {
    uint8_t current = 0;
    for(size_t b = 0; b < precision; b++) {
      if(vec[b].contains(i)) {
        current = current | (1 << b);
      }
    }
    ret[i] = current;
  }
}

/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
BitSerialMatrix toBitSerialMatrix(const uint8_t * mat, const size_t rows, const size_t cols, size_t bits) {
  BitSerialMatrix ret;
  for(size_t r = 0; r < rows; r++) {
    BitSerialVector current = toBitSerialVector(&mat[r*cols], cols, bits);
    ret.push_back(current);
  }
  return ret;
}

/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
void fromBitSerialMatrix(const BitSerialMatrix & mat, uint8_t * ret) {
  const size_t rows = mat.size();
  const size_t bits = mat[0].size();
  const size_t cols = mat[0][0].size();

  for(size_t r = 0; r < rows; r++) {
    fromBitSerialVector(mat[r], &ret[r*cols]);
  }
}

/**
* Convert a numpy array into different gemm-bitserial formats
*/

BitSerialVector toBitSerialVector(const cnpy::NpyArray & vec, const size_t bits) {
  assert(vec.word_size == sizeof(uint8_t));
  assert(vec.shape.size() == 1);
  size_t n = vec.shape[0];
  return toBitSerialVector((const uint8_t *)vec.data, n, bits);
}

BitSerialMatrix toBitSerialMatrix(const cnpy::NpyArray & mat, size_t bits) {
  assert(mat.word_size == sizeof(uint8_t));
  assert(mat.shape.size() == 2);
  assert(mat.fortran_order == false);
  size_t rows = mat.shape[0];
  size_t cols = mat.shape[1];
  return toBitSerialMatrix((const uint8_t *) mat.data, rows, cols, bits);
}

ThresholdMatrix toThresholdMatrix(const cnpy::NpyArray & mat) {
  assert(mat.word_size == sizeof(AccumulateElem));
  assert(mat.shape.size() == 2);
  assert(mat.fortran_order == false);
  size_t num_thres_levels = mat.shape[0];
  size_t num_channels = mat.shape[1];
  AccumulateElem * dataptr = (AccumulateElem *) mat.data;
  ThresholdMatrix ret;
  for(size_t t = 0; t < num_thres_levels; t++) {
    AccumulateVector ctl;
    for(size_t c = 0; c < num_channels; c++) {
      ctl.push_back(dataptr[t * num_channels + c]);
    }
    ret.push_back(ctl);
  }
  return ret;
}

FloatVector toFloatVector(const cnpy::NpyArray & vec) {
  assert(vec.word_size == sizeof(float));
  assert(vec.shape.size() == 1);
  size_t n = vec.shape[0];
  FloatVector ret;
  float * dataptr = (float *) vec.data;
  for(size_t i = 0; i < n; i++) {
    ret.push_back(dataptr[i]);
  }
  return ret;
}
