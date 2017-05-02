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
void fromBitSerialVector(const BitSerialVector & vec, const size_t n, uint8_t * ret) {
  const size_t bits = vec.size();
  for(size_t i = 0; i < n; i++) {
    uint8_t current = 0;
    for(size_t b = 0; b < bits; b++) {
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
void fromBitSerialMatrix(const BitSerialMatrix & mat, const size_t rows, const size_t cols, size_t bits, uint8_t * ret) {
  for(size_t r = 0; r < rows; r++) {
    fromBitSerialVector(mat[r], cols, &ret[r*cols]);
  }
}
