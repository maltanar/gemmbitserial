#include "gemm-bitserial.h"
#include <iostream>
using namespace std;

std::ostream &operator<<(std::ostream &os, BitVector const &m) {
  os << "BitVector of " << m.size() << " bits : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << m.contains(i) << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, BitSerialVector const &m) {
  os << "BitVector of " << m.size() << " bits of precision " << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    cout << "Bit position " << i << ":" << endl;
    os << m[i] << endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, ResultVector const &m) {
  os << "ResultVector of " << m.size() << " elements : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << (int) m[i] << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, AccumulateVector const &m) {
  os << "AccumulateVector of " << m.size() << " elements : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << (int) m[i] << std::endl;
  }
  return os;
}

/**
* Multiply a gemm-bitserial matrix and vector
*/
AccumulateVector bitSerialMatrixVector(const BitSerialMatrix & A, const BitSerialVector & x, const size_t cols, const bool Asigned, const bool xsigned) {
  const size_t rows = A.size();
  const size_t Abits = A[0].size();
  const size_t xbits = x.size();
  AccumulateVector ret(rows, 0);

  for(size_t r = 0; r < rows; r++) {
    AccumulateElem rowres = 0;
    for(size_t Abit = 0; Abit < Abits; Abit++) {
      const BitVector & car = A[r][Abit];
      for(size_t xbit = 0; xbit < xbits; xbit++) {
        // AND and popcount
        uint32_t contr = car.and_cardinality(x[xbit]);
        // scale
        contr = contr << (Abit + xbit);
        // negate if needed
        bool neg_A = Asigned && (Abit == Abits-1);
        bool neg_x = xsigned && (xbit == xbits-1);
        bool neg = neg_A ^ neg_x;
        rowres += neg ? -contr : contr;
      }
    }
    ret[r] = rowres;
  }
  return ret;
}

/**
* Multiply two gemm-bitserial matrices. Assumes B is transposed. The result is
* also produced in transposed format.
*/
AccumulateMatrix bitSerialMatrixMatrix(const BitSerialMatrix & A, const BitSerialMatrix & B, const size_t cols, const bool Asigned, const bool Bsigned) {
  // TODO this is a naive matmul implementation, matrix-vector with one column of B at a time
  // to get higher performance, we should at least have stripes
  const size_t out_rows = A.size();
  const size_t out_cols = B.size();
  const size_t Abits = A[0].size();
  const size_t Bbits = B[0].size();
  AccumulateMatrix ret;

  for(size_t o = 0; o < out_cols; o++) {
    ret.push_back(bitSerialMatrixVector(A, B[o], cols, Asigned, Bsigned));
  }

  return ret;
}

ResultVector bitSerialMatrixVectorThreshold(const BitSerialMatrix & A, const BitSerialVector & x, const ThresholdMatrix & T, const size_t cols,  const bool Asigned, const bool xsigned) {
  // this could have been implemented by just calling the matrix-vector first
  // then thresholding the results, but we want more instruction-level parallelism
  // to keep the CPU functional units occupied, so the matrix-vector code is
  // repeated, and the thresholding is directly inserted inside the loop.
  const size_t rows = A.size();
  const size_t Abits = A[0].size();
  const size_t xbits = x.size();
  const size_t numThres = T.size();
  const size_t numThresChans = T[0].size();
  ResultVector ret;
  ResultElem postthres;
  for(size_t r = 0; r < rows; r++) {
    AccumulateElem rowres = 0;
    for(size_t Abit = 0; Abit < Abits; Abit++) {
      for(size_t xbit = 0; xbit < xbits; xbit++) {
        // AND and popcount
        uint32_t contr = A[r][Abit].and_cardinality(x[xbit]);
        // scale
        contr = contr << (Abit + xbit);
        // negate if needed
        bool neg_A = Asigned && (Abit == Abits-1);
        bool neg_x = xsigned && (xbit == xbits-1);
        bool neg = neg_A ^ neg_x;
        rowres += neg ? -contr : contr;
      }
    }
    // handle both broadcast and one-to-one threshold channel cases
    postthres = 0;
    for(size_t t = 0; t < numThres; t++) {
      if(numThresChans == rows) {
        // one threshold channel for each row
        postthres += (rowres >= T[t][r]) ? 1 : 0;
      } else {
        // cycle threshold channels
        // TODO get rid of modulo here for higher performance
        postthres += (rowres >= T[t][r % numThresChans]) ? 1 : 0;
      }
    }
    ret.push_back(postthres);
  }
  return ret;
}

/**
* Apply a set of thresholds to an AccumulateVector, returning the number of crossed thresholds
*/
ResultVector threshold(const AccumulateVector & x, const ThresholdMatrix & T) {
  const size_t rows = x.size();
  const size_t numThres = T.size();
  const size_t numThresChans = T[0].size();
  ResultVector ret;
  ResultElem postthres;

  for(size_t r = 0; r < rows; r++) {
    postthres = 0;
    for(size_t t = 0; t < numThres; t++) {
      if(numThresChans == rows) {
        // one threshold channel for each row
        postthres += (x[r] >= T[t][r]) ? 1 : 0;
      } else {
        // cycle threshold channels
        // TODO get rid of modulo here for higher performance
        postthres += (x[r] >= T[t][r % numThresChans]) ? 1 : 0;
      }
    }
    ret.push_back(postthres);
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
