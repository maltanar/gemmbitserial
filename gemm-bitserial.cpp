#include "gemm-bitserial.h"

/**
* Convert a buffer of unsigned char values into a gemm-bitserial vector
*/
BitSerialVector toBitSerialVector(const uint8_t * vec, const size_t n, const size_t bits) {
  BitSerialVector ret;

  for(size_t b = 0; b < bits; b++) {
    Roaring currentBitGroup;
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

/**
* Multiply a gemm-bitserial matrix and vector
*/
ResultVector bitSerialMatrixVector(const BitSerialMatrix & A, const BitSerialVector & x, const size_t cols, const bool Asigned, const bool xsigned) {
  const size_t rows = A.size();
  const size_t Abits = A[0].size();
  const size_t xbits = x.size();
  ResultVector ret;

  for(size_t r = 0; r < rows; r++) {
    ResultElem rowres = 0;
    BitSerialVector crow = A[r];
    for(size_t Abit = 0; Abit < Abits; Abit++) {
      for(size_t xbit = 0; xbit < xbits; xbit++) {
        // AND and popcount
        uint32_t contr = crow[Abit].and_cardinality(x[xbit]);
        // scale
        contr = contr << (Abit + xbit);
        // negate if needed
        bool neg_A = Asigned && (Abit == Abits-1);
        bool neg_x = xsigned && (xbit == xbits-1);
        bool neg = neg_A ^ neg_x;
        rowres += neg ? -contr : contr;
      }
    }
    ret.push_back(rowres);
  }
  return ret;
}

/**
* Generate a random vector with given dimension and number of bits <= 8
*/
void generateRandomVector(size_t bits, size_t dim, uint8_t * ret) {
  uint8_t minVal = 0;
  uint8_t maxVal = (1 << bits) - 1;
  for(size_t i = 0; i < dim; i++) {
    ret[i] = rand() % maxVal;
  }
}
