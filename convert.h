#pragma once
#include "bitvector.h"
#include "cnpy.h"
typedef MyBitVector BitVector;
typedef std::vector<BitVector> BitSerialVector;
typedef std::vector<BitSerialVector> BitSerialMatrix;
typedef int32_t AccumulateElem;
typedef std::vector<AccumulateElem> AccumulateVector;
typedef std::vector<AccumulateVector> AccumulateMatrix;
typedef uint8_t ResultElem;
typedef std::vector<ResultElem> ResultVector;
typedef std::vector<AccumulateVector> ThresholdMatrix;
typedef std::vector<float> FloatVector;
/**
* Convert a buffer of unsigned char values into a gemm-bitserial vector
*/
BitSerialVector toBitSerialVector(const uint8_t * vec, const size_t n, const size_t bits);

/**
* Convert a std::vector of values into a gemm-bitserial vector. Values will be cast to
* uint8_t first.
*/
template <typename T>
BitSerialVector toBitSerialVector(const std::vector<T> & vec, const size_t bits) {
  const size_t n = vec.size();
  uint8_t * buf = new uint8_t[n];
  for(size_t i = 0; i < n; i++) {
    buf[i] = (uint8_t) vec[i];
  }
  BitSerialVector ret = toBitSerialVector(buf, n, bits);
  delete [] buf;
  return ret;
}

/**
* Convert a buffer of values into a gemm-bitserial vector. Values will be cast to
* uint8_t first.
*/
template <typename T>
BitSerialVector toBitSerialVector(const T * vec, const size_t n, const size_t bits) {
  uint8_t * buf = new uint8_t[n];
  for(size_t i = 0; i < n; i++) {
    buf[i] = (uint8_t) vec[i];
  }
  BitSerialVector ret = toBitSerialVector(buf, n, bits);
  delete [] buf;
  return ret;
}

/**
* Convert a gemm-bitserial vector into a buffer of unsigned char values
*/
void fromBitSerialVector(const BitSerialVector & vec, uint8_t * ret);

/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
BitSerialMatrix toBitSerialMatrix(const uint8_t * mat, const size_t rows, const size_t cols, size_t bits);

template <typename T>
BitSerialMatrix toBitSerialMatrix(const std::vector<T> & vec, const size_t rows, const size_t cols, const size_t bits) {
  const size_t n = vec.size();
  uint8_t * buf = new uint8_t[n];
  for(size_t i = 0; i < n; i++) {
    buf[i] = (uint8_t) vec[i];
  }
  BitSerialMatrix ret = toBitSerialMatrix(buf, rows, cols, bits);
  delete [] buf;
  return ret;
}

template <typename T>
BitSerialMatrix toBitSerialMatrix(const T* vec, const size_t rows, const size_t cols, const size_t bits) {
  const size_t n = rows*cols;
  uint8_t * buf = new uint8_t[n];
  for(size_t i = 0; i < n; i++) {
    buf[i] = (uint8_t) vec[i];
  }
  BitSerialMatrix ret = toBitSerialMatrix(buf, rows, cols, bits);
  delete [] buf;
  return ret;
}


/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
void fromBitSerialMatrix(const BitSerialMatrix & mat, size_t bits, uint8_t * ret);


/**
* Convert a numpy array into different gemm-bitserial formats
*/
BitSerialVector toBitSerialVector(const cnpy::NpyArray & vec, const size_t bits);
BitSerialMatrix toBitSerialMatrix(const cnpy::NpyArray & mat, size_t bits);
ThresholdMatrix toThresholdMatrix(const cnpy::NpyArray & mat);
FloatVector toFloatVector(const cnpy::NpyArray & vec);
