#pragma once
#include <math.h>

// generic (non-architecture-specific) implementations of gemmBitserial
// and other related functions

// Utility function to increment-and-align "in" to "af"
inline uint64_t alignTo(uint64_t in, uint64_t af) {
  return in + (af - (in % af));
}

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

/* Multiply a lhs_block x rhs_block chunk of the given matrices, starting at
  (bA, bBT) using 2x1x2 register tiling. For internal use.
*/
template <typename AccType>
inline void gemmBinary_generic_chunk_tile2x1x2(uint64_t * A, uint64_t * BT, AccType * CT,
  AccType alpha, uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t bA, uint64_t bBT, uint64_t lhs_block, uint64_t rhs_block) {
  const uint64_t Atile = 2, DepthTile = 1, BTtile = 2;
  const size_t num_acc = Atile*BTtile;

  for(uint64_t rBT = bBT; rBT < bBT + rhs_block; rBT += BTtile) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = bA; rA < bA + lhs_block; rA += Atile) {
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

/* Multiply A and BT using 2x1x2 register tiling. For internal use.
*/
template <typename AccType>
inline void gemmBinary_generic_tile2x1x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
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

/* Multiply a lhs_block x rhs_block chunk of the given matrices, starting at
  (bA, bBT) without any register tiling. For internal use.
*/
template <typename AccType>
inline void gemmBinary_generic_chunk_naive(uint64_t * A, uint64_t * BT, AccType * CT,
  AccType alpha, uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t bA, uint64_t bBT, uint64_t lhs_block, uint64_t rhs_block) {

  for(uint64_t rBT = bBT; rBT < bBT + rhs_block; rBT++) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = bA; rA < bA + lhs_block; rA++) {
      uint64_t * Aptr = &A[rA * depth_words];
      AccType acc = 0;
      for(uint64_t d = 0; d < depth_words; d++) {
        acc += __builtin_popcountll(Aptr[d] & BTptr[d]);
      }
      CT[rBT * rowsA + rA ] += acc * alpha;
    }
  }
}

/* CT = A * BT using cache blocking and 2x1x2 register blocking where possible.
   For internal use.
*/
template <typename AccType>
void gemmBinary_generic_L1_tile2x1x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  // TODO get block sizes as params, or memoize
  float cacheBits = 32 * 1024 * 8;
  // register blocking factors
  const uint64_t Atile = 2, DepthTile = 1, BTtile = 2;
  uint64_t lhs_block, rhs_block;
  // find L1 block size
  findBlockSize(cacheBits, depth_words*64, (float) Atile, (float) BTtile, sizeof(AccType)*8, lhs_block, rhs_block);
  assert(lhs_block % Atile == 0);
  assert(depth_words % DepthTile == 0);
  assert(rhs_block % BTtile == 0);

  uint64_t rowsA_aligned = alignTo(rowsA, lhs_block);
  uint64_t rowsBT_aligned = alignTo(rowsBT, rhs_block);

  for(uint64_t bBT = 0; bBT < rowsBT; ) {
    uint64_t BT_block = (rowsBT - bBT) < rhs_block ? (rowsBT - bBT) : rhs_block;
    for(uint64_t bA = 0; bA < rowsA; ) {
      uint64_t A_block = (rowsA - bA) < lhs_block ? (rowsA - bA) : lhs_block;
      // dispatch appropriate chunk function based on block size
      if(A_block == lhs_block && BT_block == rhs_block) {
        gemmBinary_generic_chunk_tile2x1x2(
          A, BT, CT, alpha, rowsA, depth_words, rowsBT, bA, bBT, A_block, BT_block
        );
      } else {
        gemmBinary_generic_chunk_naive(
          A, BT, CT, alpha, rowsA, depth_words, rowsBT, bA, bBT, A_block, BT_block
        );
      }
      bA += A_block;
    }
    bBT += BT_block;
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
      // TODO dispatch according to matrix dimensions
      // TODO compute block shapes only once
      gemmBinary_generic_L1_tile2x1x2(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
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
