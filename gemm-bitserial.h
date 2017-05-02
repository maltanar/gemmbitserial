#pragma once
#include <stdint.h>
#include <string.h>
#include "bitvector.h"
#include "convert.h"
#include <ostream>

/**
* Print contents of different vector types
*/
std::ostream &operator<<(std::ostream &os, BitVector const &m);
std::ostream &operator<<(std::ostream &os, BitSerialVector const &m);
std::ostream &operator<<(std::ostream &os, ResultVector const &m);
std::ostream &operator<<(std::ostream &os, AccumulateVector const &m);

/**
* Multiply a gemm-bitserial matrix and vector
*/
AccumulateVector bitSerialMatrixVector(const BitSerialMatrix & A, const BitSerialVector & x, const size_t cols, const bool Asigned = false, const bool xsigned = false);

/**
* Multiply two gemm-bitserial matrices. Assumes B is transposed. The result is
* also produced in transposed format.
*/
AccumulateMatrix bitSerialMatrixMatrix(const BitSerialMatrix & A, const BitSerialMatrix & B, const size_t cols, const bool Asigned = false, const bool Bsigned = false);

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
