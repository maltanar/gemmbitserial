#pragma once
#include <stdint.h>
#include <string.h>
#include <vector>
#include <memory>

class MyBitVector {
public:
  MyBitVector(size_t numBits);

  void add(uint64_t index);
  void clear();
  bool contains(uint64_t index) const;
  uint64_t and_cardinality(const MyBitVector & rhs) const;

protected:
  std::vector<uint64_t> m_buf;
  size_t m_numWords, m_bufWordBits;
  uint64_t m_posShift, m_posMask;

  uint64_t getWPos(uint64_t bitIndex) const;
  uint64_t getBPos(uint64_t bitIndex) const;
};

typedef MyBitVector BitVector;
typedef std::vector<BitVector> BitSerialVector;
typedef std::vector<BitSerialVector> BitSerialMatrix;
typedef int32_t ResultElem;
typedef std::vector<ResultElem> ResultVector;

/**
* Convert a buffer of unsigned char values into a gemm-bitserial vector
*/
BitSerialVector toBitSerialVector(const uint8_t * vec, const size_t n, const size_t bits);

/**
* Convert a gemm-bitserial vector into a buffer of unsigned char values
*/
void fromBitSerialVector(const BitSerialVector & vec, const size_t n, uint8_t * ret);

/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
BitSerialMatrix toBitSerialMatrix(const uint8_t * mat, const size_t rows, const size_t cols, size_t bits);

/**
* Convert a buffer of unsigned char values into a gemm-bitserial matrix
*/
void fromBitSerialMatrix(const BitSerialMatrix & mat, const size_t rows, const size_t cols, size_t bits, uint8_t * ret);

/**
* Multiply a gemm-bitserial matrix and vector
*/
ResultVector bitSerialMatrixVector(const BitSerialMatrix & A, const BitSerialVector & x, const size_t cols, const bool Asigned = false, const bool xsigned = false);

/**
* Generate a random vector with given dimension and number of bits <= 8
*/
void generateRandomVector(size_t bits, size_t dim, uint8_t * ret);
