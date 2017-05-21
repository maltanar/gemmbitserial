#pragma once
#include <arm_neon.h>

// architecture-specific implementations of primitives for ARM NEON

template <typename AccType>
void gemmBinary_neon_stripe2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  assert(rowsA % 2 == 0);
  assert(rowsBT % 2 == 0);

  for(uint64_t rBT = 0; rBT < rowsBT; rBT+=2) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA+=2) {
      uint64_t * Aptr = &A[rA * depth_words];
      uint64_t acc[4] = {0};
      uint8x8_t acc_neon[4];
      // TODO could keep things in 16-bit accumulators for perf.
      uint64x1_t acc2_neon[4];
      // initialize accumulators to zero
      for(int init = 0; init < 4; init++) {
        acc_neon[init] = vcreate_u8(0);
        acc2_neon[init] = vcreate_u64(0);
      }

      for(uint64_t d = 0; d < depth_words; d++) {
        uint8x8_t a0, b0, a1, b1;
        a0 = vld1_u8((uint8_t *) &Aptr[d]);
        a1 = vld1_u8((uint8_t *) &Aptr[d + depth_words]);
        b0 = vld1_u8((uint8_t *) &BTptr[d]);
        b1 = vld1_u8((uint8_t *) &BTptr[d + depth_words]);

        acc_neon[0] = vadd_u8(acc_neon[0], vcnt_u8(vand_u8(a0, b0)));
        acc_neon[1] = vadd_u8(acc_neon[1], vcnt_u8(vand_u8(a0, b1)));
        acc_neon[2] = vadd_u8(acc_neon[2], vcnt_u8(vand_u8(a1, b0)));
        acc_neon[3] = vadd_u8(acc_neon[3], vcnt_u8(vand_u8(a1, b1)));

        if(d == 30 || d == depth_words-1) {
          /* hsum over 8-bit accumulators when end or overflow*/
          for(int init = 0; init < 4; init++) {
            acc2_neon[init] = vadd_u64(acc2_neon[init], vpaddl_u32(vpaddl_u16(vpaddl_u8(acc_neon[init]))));
            acc_neon[init] = vcreate_u8(0);
          }
        }
      }
      /* move into regular accumulators */
      for(int init = 0; init < 4; init++) {
        vst1_u64(&acc[init], acc2_neon[init]);
      }

      CT[rBT * rowsA + rA] += acc[0] * alpha;
      CT[(rBT + 1) * rowsA + rA] += acc[1] * alpha;
      CT[rBT * rowsA + rA + 1] += acc[2] * alpha;
      CT[(rBT + 1) * rowsA + rA + 1] += acc[3] * alpha;
    }
  }
}

template <typename AccType>
void gemmBinary_neon_tile2x2x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  assert(rowsA % 2 == 0);
  assert(rowsBT % 2 == 0);
  assert(depth_words % 2 == 0);

  for(uint64_t rBT = 0; rBT < rowsBT; rBT+=2) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA+=2) {
      uint64_t * Aptr = &A[rA * depth_words];
      uint64_t acc[4] = {0};
      uint8x16_t acc_neon[4];
      // TODO could keep things in 16-bit accumulators for perf.
      uint64x2_t acc2_neon[4];
      // initialize accumulators to zero
      for(int init = 0; init < 4; init++) {
        acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        acc2_neon[init] = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
      }

      for(uint64_t d = 0; d < depth_words; d+=2) {
        uint8x16_t a0, b0, a1, b1;
        a0 = vld1q_u8((uint8_t *) &Aptr[d]);
        a1 = vld1q_u8((uint8_t *) &Aptr[d + depth_words]);
        b0 = vld1q_u8((uint8_t *) &BTptr[d]);
        b1 = vld1q_u8((uint8_t *) &BTptr[d + depth_words]);

        acc_neon[0] = vaddq_u8(acc_neon[0], vcntq_u8(vandq_u8(a0, b0)));
        acc_neon[1] = vaddq_u8(acc_neon[1], vcntq_u8(vandq_u8(a0, b1)));
        acc_neon[2] = vaddq_u8(acc_neon[2], vcntq_u8(vandq_u8(a1, b0)));
        acc_neon[3] = vaddq_u8(acc_neon[3], vcntq_u8(vandq_u8(a1, b1)));

        if(d == 30 || d == depth_words-2) {
          /* hsum over 8-bit accumulators when end or overflow*/
          for(int init = 0; init < 4; init++) {
            acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
            acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
          }
        }
      }
      /* move into regular accumulators */
      for(int init = 0; init < 4; init++) {
        uint64_t tmp[2];
        vst1q_u64(tmp, acc2_neon[init]);
        acc[init] = tmp[0] + tmp[1];
      }

      CT[rBT * rowsA + rA] += acc[0] * alpha;
      CT[(rBT + 1) * rowsA + rA] += acc[1] * alpha;
      CT[rBT * rowsA + rA + 1] += acc[2] * alpha;
      CT[(rBT + 1) * rowsA + rA + 1] += acc[3] * alpha;
    }
  }
}


template <typename AccType>
void gemmBinary_neon_tile2x2x4(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  assert(rowsA % 2 == 0);
  assert(rowsBT % 2 == 0);
  assert(depth_words % 4 == 0);

  for(uint64_t rBT = 0; rBT < rowsBT; rBT+=2) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA+=2) {
      uint64_t * Aptr = &A[rA * depth_words];
      uint64_t acc[4] = {0};
      uint8x16_t acc_neon[4];
      // TODO could keep things in 16-bit accumulators for perf.
      uint64x2_t acc2_neon[4];
      // initialize accumulators to zero
      for(int init = 0; init < 4; init++) {
        acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        acc2_neon[init] = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
      }

      for(uint64_t d = 0; d < depth_words; d+=4) {
        uint8x16x2_t a0, b0, a1, b1;
        a0 = vld2q_u8((uint8_t *) &Aptr[d]);
        a1 = vld2q_u8((uint8_t *) &Aptr[d + depth_words]);
        b0 = vld2q_u8((uint8_t *) &BTptr[d]);
        b1 = vld2q_u8((uint8_t *) &BTptr[d + depth_words]);

        acc_neon[0] = vaddq_u8(acc_neon[0], vcntq_u8(vandq_u8(a0.val[0], b0.val[0])));
        acc_neon[0] = vaddq_u8(acc_neon[0], vcntq_u8(vandq_u8(a0.val[1], b0.val[1])));

        acc_neon[1] = vaddq_u8(acc_neon[1], vcntq_u8(vandq_u8(a0.val[0], b1.val[0])));
        acc_neon[1] = vaddq_u8(acc_neon[1], vcntq_u8(vandq_u8(a0.val[1], b1.val[1])));

        acc_neon[2] = vaddq_u8(acc_neon[2], vcntq_u8(vandq_u8(a1.val[0], b0.val[0])));
        acc_neon[2] = vaddq_u8(acc_neon[2], vcntq_u8(vandq_u8(a1.val[1], b0.val[1])));

        acc_neon[3] = vaddq_u8(acc_neon[3], vcntq_u8(vandq_u8(a1.val[0], b1.val[0])));
        acc_neon[3] = vaddq_u8(acc_neon[3], vcntq_u8(vandq_u8(a1.val[1], b1.val[1])));

        if(d == 30 || d == depth_words-4) {
          /* hsum over 8-bit accumulators when end or overflow*/
          for(int init = 0; init < 4; init++) {
            acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
            acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
          }
        }
      }
      /* move into regular accumulators */
      for(int init = 0; init < 4; init++) {
        uint64_t tmp[2];
        vst1q_u64(tmp, acc2_neon[init]);
        acc[init] = tmp[0] + tmp[1];
      }

      CT[rBT * rowsA + rA] += acc[0] * alpha;
      CT[(rBT + 1) * rowsA + rA] += acc[1] * alpha;
      CT[rBT * rowsA + rA + 1] += acc[2] * alpha;
      CT[(rBT + 1) * rowsA + rA + 1] += acc[3] * alpha;
    }
  }
}

/* Bit-serial GEMM via a series of calls to gemmBinary.
   Note that rhs must be given in transposed form, and the result is also
   produced transposed.
*/
template <typename AccType>
void gemmBitSerial_neon_usingBinary(BitSerialMatrix * lhs, BitSerialMatrix * rhs, AccType * res) {
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
      if(lhs->nrows % 2 == 0 && rhs->nrows % 2 == 0 && lhs->wordsPerRow() % 4 == 0) {
        gemmBinary_neon_tile2x2x4(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      }
      else if(lhs->nrows % 2 == 0 && rhs->nrows % 2 == 0 && lhs->wordsPerRow() % 2 == 0) {
        gemmBinary_neon_tile2x2x2(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      }
      else if(lhs->nrows % 2 == 0 && rhs->nrows % 2 == 0) {
        gemmBinary_neon_stripe2(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      } else {
        gemmBinary_generic_naive(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
      }
    }
  }
}
