#pragma once
#include <arm_neon.h>

// architecture-specific implementations of primitives for ARM NEON
// the tiling strategies are named as:
// _tile<lhs-rows><depth-elems><rhs-rows>

template <typename AccType>
inline void gemmBinary_neon_chunk_tile4x2x2(uint64_t * A, uint64_t * BT, AccType * CT,
  AccType alpha, uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t bA, uint64_t bBT, uint64_t lhs_block, uint64_t rhs_block) {
  const uint64_t Atile = 4, DepthTile = 2, BTtile = 2;
  const size_t num_acc = Atile*BTtile;

  for(uint64_t rBT = bBT; rBT < bBT + rhs_block; rBT += BTtile) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = bA; rA < bA + lhs_block; rA += Atile) {
      uint64_t * Aptr = &A[rA * depth_words];
      uint64_t acc[num_acc] = {0};
      uint8x16_t acc_neon[num_acc];
      // TODO could keep things in 16-bit accumulators for perf.
      uint64x2_t acc2_neon[num_acc];
      // initialize accumulators to zero
      for(size_t init = 0; init < num_acc; init++) {
        acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        acc2_neon[init] = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
      }

      for(uint64_t d = 0; d < depth_words; d += DepthTile) {
        uint8x16_t a0, b0, a1, b1, a2, a3;

        a0 = vld1q_u8((uint8_t *) &Aptr[d + 0*depth_words]);
        a1 = vld1q_u8((uint8_t *) &Aptr[d + 1*depth_words]);
        a2 = vld1q_u8((uint8_t *) &Aptr[d + 2*depth_words]);
        a3 = vld1q_u8((uint8_t *) &Aptr[d + 3*depth_words]);
        b0 = vld1q_u8((uint8_t *) &BTptr[d]);
        b1 = vld1q_u8((uint8_t *) &BTptr[d + depth_words]);

        acc_neon[0] = vaddq_u8(acc_neon[0], vcntq_u8(vandq_u8(a0, b0)));
        acc_neon[1] = vaddq_u8(acc_neon[1], vcntq_u8(vandq_u8(a0, b1)));
        acc_neon[2] = vaddq_u8(acc_neon[2], vcntq_u8(vandq_u8(a1, b0)));
        acc_neon[3] = vaddq_u8(acc_neon[3], vcntq_u8(vandq_u8(a1, b1)));
        acc_neon[4] = vaddq_u8(acc_neon[4], vcntq_u8(vandq_u8(a2, b0)));
        acc_neon[5] = vaddq_u8(acc_neon[5], vcntq_u8(vandq_u8(a2, b1)));
        acc_neon[6] = vaddq_u8(acc_neon[6], vcntq_u8(vandq_u8(a3, b0)));
        acc_neon[7] = vaddq_u8(acc_neon[7], vcntq_u8(vandq_u8(a3, b1)));

        if((d & 7L) == 7L) {
          // hsum over 8-bit accumulators when end or overflow
          for(size_t init = 0; init < num_acc; init++) {
            acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
            acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
          }
        }
      }
      /* move into regular accumulators */
      for(size_t init = 0; init < num_acc; init++) {
        uint64_t tmp[2];
        acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
        vst1q_u64(tmp, acc2_neon[init]);
        acc[init] = tmp[0] + tmp[1];
      }

      for(uint64_t at = 0; at < Atile; at++) {
        for(uint64_t bt = 0; bt < BTtile; bt++) {
          CT[(rBT + bt) * rowsA + (rA + at)] += acc[at * BTtile + bt] * alpha;
        }
      }
    }
  }
}

template <typename AccType>
void gemmBinary_neon_tile4x2x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  const uint64_t Atile = 4, BTtile = 2, DepthTile = 2;
  const size_t num_acc = Atile*BTtile;
  assert(rowsA % Atile == 0);
  assert(depth_words % DepthTile == 0);
  assert(rowsBT % BTtile == 0);

  for(uint64_t rBT = 0; rBT < rowsBT; rBT += BTtile) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA += Atile) {
      uint64_t * Aptr = &A[rA * depth_words];
      uint64_t acc[num_acc] = {0};
      uint8x16_t acc_neon[num_acc];
      // TODO could keep things in 16-bit accumulators for perf.
      uint64x2_t acc2_neon[num_acc];
      // initialize accumulators to zero
      for(size_t init = 0; init < num_acc; init++) {
        acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
        acc2_neon[init] = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
      }

      for(uint64_t d = 0; d < depth_words; d += DepthTile) {
        uint8x16_t a0, b0, a1, b1, a2, a3;

        a0 = vld1q_u8((uint8_t *) &Aptr[d + 0*depth_words]);
        a1 = vld1q_u8((uint8_t *) &Aptr[d + 1*depth_words]);
        a2 = vld1q_u8((uint8_t *) &Aptr[d + 2*depth_words]);
        a3 = vld1q_u8((uint8_t *) &Aptr[d + 3*depth_words]);
        b0 = vld1q_u8((uint8_t *) &BTptr[d]);
        b1 = vld1q_u8((uint8_t *) &BTptr[d + depth_words]);

        acc_neon[0] = vaddq_u8(acc_neon[0], vcntq_u8(vandq_u8(a0, b0)));
        acc_neon[1] = vaddq_u8(acc_neon[1], vcntq_u8(vandq_u8(a0, b1)));
        acc_neon[2] = vaddq_u8(acc_neon[2], vcntq_u8(vandq_u8(a1, b0)));
        acc_neon[3] = vaddq_u8(acc_neon[3], vcntq_u8(vandq_u8(a1, b1)));
        acc_neon[4] = vaddq_u8(acc_neon[4], vcntq_u8(vandq_u8(a2, b0)));
        acc_neon[5] = vaddq_u8(acc_neon[5], vcntq_u8(vandq_u8(a2, b1)));
        acc_neon[6] = vaddq_u8(acc_neon[6], vcntq_u8(vandq_u8(a3, b0)));
        acc_neon[7] = vaddq_u8(acc_neon[7], vcntq_u8(vandq_u8(a3, b1)));

        if((d & 7L) == 7L) {
          // hsum over 8-bit accumulators when end or overflow
          for(size_t init = 0; init < num_acc; init++) {
            acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
            acc_neon[init] = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
          }
        }
      }
      /* move into regular accumulators */
      for(size_t init = 0; init < num_acc; init++) {
        uint64_t tmp[2];
        acc2_neon[init] = vaddq_u64(acc2_neon[init], vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon[init]))));
        vst1q_u64(tmp, acc2_neon[init]);
        acc[init] = tmp[0] + tmp[1];
      }

      for(uint64_t at = 0; at < Atile; at++) {
        for(uint64_t bt = 0; bt < BTtile; bt++) {
          CT[(rBT + bt) * rowsA + (rA + at)] += acc[at * BTtile + bt] * alpha;
        }
      }
    }
  }
}


/* CT = A * BT using cache blocking and 2x1x2 register blocking where possible.
   For internal use.
*/
template <typename AccType>
void gemmBinary_neon_L1_tile4x2x2(uint64_t * A, uint64_t * BT, AccType * CT, AccType alpha,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT) {
  // TODO get block sizes as params, or memoize
  float cacheBits = 16 * 1024 * 8;
  // register blocking factors
  const uint64_t Atile = 4, DepthTile = 2, BTtile = 2;
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
        gemmBinary_neon_chunk_tile4x2x2(
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
      gemmBinary_neon_L1_tile4x2x2(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, alpha, lhs->nrows, lhs->wordsPerRow(), rhs->nrows);
    }
  }
}
