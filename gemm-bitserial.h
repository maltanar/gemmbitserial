#pragma once
#include <stdint.h>
#include <string.h>
#include "bitvector.h"

typedef MyBitVector BitVector;
typedef std::vector<BitVector> BitSerialVector;
typedef std::vector<BitSerialVector> BitSerialMatrix;
typedef int32_t AccumulateElem;
typedef std::vector<AccumulateElem> AccumulateVector;
typedef uint8_t ResultElem;
typedef std::vector<ResultElem> ResultVector;
typedef std::vector<AccumulateVector> ThresholdMatrix;

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
AccumulateVector bitSerialMatrixVector(const BitSerialMatrix & A, const BitSerialVector & x, const size_t cols, const bool Asigned = false, const bool xsigned = false);

/**
* Multiply a gemm-bitserial matrix and vector, followed by a thresholding operation
*/
ResultVector bitSerialMatrixVectorThreshold(const BitSerialMatrix & A, const BitSerialVector & x, const ThresholdMatrix & T, const size_t cols,  const bool Asigned = false, const bool xsigned = false);

/**
* Apply a set of thresholds to an AccumulateVector, returning the number of crossed thresholds
*/
ResultVector threshold(const AccumulateVector & x, const ThresholdMatrix & T);

/**
* Generate a random vector with given dimension and number of bits <= 8
*/
void generateRandomVector(size_t bits, size_t dim, uint8_t * ret);
