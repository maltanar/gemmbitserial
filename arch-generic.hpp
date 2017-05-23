#pragma once
#include <math.h>

// generic (non-architecture-specific) implementations of gemmBitserial
// and other related functions


/* Utility function to find block size under the following assumptions:
   - size of lhs block + rhs block + result block <= cacheBits
   - no blocking along depth (i.e. only entire rows of dBits bits)
   - lhsMult and rhsMult determine the ratio for lhs and rhs rows in cache
   - returned lhsRows and rhsRows are divisible by lhsMult and rhsMult, respectively
   - each result elem takes bitsPerRes bits
*/
void findBlockSize(float cacheBits, float dBits, float lhsMult, float rhsMult,
  float bitsPerRes, uint64_t & lhsRows, uint64_t & rhsRows) {
  float a = bitsPerRes * lhsMult * rhsMult;
  float b = dBits*(lhsMult + rhsMult);
  float c = -cacheBits;
  float discr = sqrt(b*b - 4 * a * c);
  assert(discr > 0);
  int64_t x0 = floor((-b + discr) / (2*a));
  int64_t x1 = floor((-b - discr) / (2*a));
  int64_t x = x0 > x1 ? x0 : x1;
  assert(x > 0);
  lhsRows = lhsMult * x;
  rhsRows = rhsMult * x;
}


/* Multiply two binary matrices. Note that rhs must be given in transposed
   form, and the result is also produced transposed.
*/
template <typename AccType>
void gemmBinary_generic_naive(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  for(uint64_t rBT = 0; rBT < rowsBT; rBT++) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA++) {
      uint64_t * Aptr = &A[rA * depth_words];
      AccType acc = 0;
      for(uint64_t d = 0; d < depth_words; d++) {
        acc += __builtin_popcountll(Aptr[d] & BTptr[d]);
      }
      CT[rBT * rowsA + rA] += acc * alpha;
    }
  }
}

template <typename AccType>
void gemmBinary_generic_tile2x1x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  const uint64_t Atile = 2, DepthTile = 1, BTtile = 2;
  const size_t num_acc = Atile*BTtile;
  assert(rowsA % Atile == 0);
  assert(depth_words % DepthTile == 0);
  assert(rowsBT % BTtile == 0);

  for(uint64_t rBT = 0; rBT < rowsBT; rBT += BTtile) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA += Atile) {
      uint64_t * Aptr = &A[rA * depth_words];
      AccType acc[num_acc] = {0};
      for(uint64_t d = 0; d < depth_words; d += DepthTile) {
        const uint64_t a0 = Aptr[d], a1 = Aptr[d + depth_words];
        const uint64_t b0 = BTptr[d], b1 = BTptr[d + depth_words];
        acc[0] += __builtin_popcountll(a0 & b0);
        acc[1] += __builtin_popcountll(a0 & b1);
        acc[2] += __builtin_popcountll(a1 & b0);
        acc[3] += __builtin_popcountll(a1 & b1);
      }
      for(uint64_t at = 0; at < Atile; at++) {
        for(uint64_t bt = 0; bt < BTtile; bt++) {
          CT[(rBT + bt) * rowsA + (rA + at)] += acc[at * BTtile + bt] * alpha;
        }
      }
    }
  }
}

/* Bit-serial GEMM via a series of calls to gemmBinary.
   Note that rhs must be given in transposed form, and the result is also
   produced transposed.
*/
template <typename AccType>
void gemmBitSerial_generic_usingBinary(BitSerialMatrix * lhs, BitSerialMatrix * rhs, AccType * res) {
  // ensure that matrix shapes are compatible
  assert(lhs->ncols == rhs->ncols);
  const uint64_t lhsbits = lhs->nbits;
  const uint64_t rhsbits = rhs->nbits;
  // clear contents of result matrix by setting everything to zero
  memset(res, 0, lhs->nrows*rhs->nrows*sizeof(AccType));
  // call binary GEMM for each bit position
  for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
    bool neg_lhs = lhs->issigned && (lbit == lhsbits-1);
    for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
      bool neg_rhs = rhs->issigned && (rbit == rhsbits-1);
      bool neg = neg_rhs ^ neg_lhs;
      AccType alpha = neg ? -(1 << (lbit+rbit)) : (1 << (lbit+rbit));
      if(lhs->nrows % 2 == 0 && rhs->nrows % 2 == 0) {
        gemmBinary_generic_tile2x1x2(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      } else {
        gemmBinary_generic_naive(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      }
    }
  }
}


/* Standalone bit-serial GEMM. Note that rhs must be given in transposed
   form, and the result is also produced transposed.
*/
template <typename AccType>
void gemmBitSerial_generic_naive(BitSerialMatrix * lhs, BitSerialMatrix * rhs, AccType * res) {
  // ensure that matrix shapes are compatible
  assert(lhs->ncols == rhs->ncols);
  const uint64_t lhsbits = lhs->nbits;
  const uint64_t rhsbits = rhs->nbits;
  const uint64_t out_rows = lhs->nrows;
  const uint64_t out_cols = rhs->nrows;
  const uint64_t depth = lhs->wordsPerRow();

  for(uint64_t i = 0; i < out_cols; i++) {
    for(uint64_t j = 0; j < out_rows; j++) {
      AccType rowres = 0;
      for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
        bool neg_lhs = lhs->issigned && (lbit == lhsbits-1);
        for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
          bool neg_rhs = rhs->issigned && (rbit == rhsbits-1);
          uint64_t * ldata = lhs->rowptr(lbit, j);
          uint64_t * rdata = rhs->rowptr(rbit, i);
          uint64_t andcard = 0;
          // AND-popcount-accumulate over row pair
          for(uint64_t k = 0; k < depth; k++) {
            andcard += __builtin_popcountll(ldata[k] & rdata[k]);
          }
          // scale
          andcard = andcard << (lbit + rbit);
          // negate if needed
          rowres += (neg_lhs ^ neg_rhs) ? -andcard : andcard;
        }
      }
      res[i * out_rows + j] = rowres;
    }
  }
}
