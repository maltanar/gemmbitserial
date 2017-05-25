#pragma once
#include <arm_neon.h>

// architecture-specific implementations of primitives for ARM NEON
// the tiling strategies are named as:
// _tile<lhs-rows><depth-elems><rhs-rows>

static GEMMContext allocGEMMContext_neon(
  uint64_t lhsRows, uint64_t depth, uint64_t rhsRows,
  uint64_t lhsBits, uint64_t rhsBits,
  bool lhsSigned, bool rhsSigned
) {
  const uint64_t regblock_lhs = 4;
  const uint64_t regblock_d = 2;
  const uint64_t regblock_rhs = 2;
  const uint64_t cacheBits = 16*1024*8;

  return allocGEMMContext_base(
    lhsRows, depth, rhsRows, lhsBits, rhsBits, lhsSigned, rhsSigned,
    regblock_lhs, regblock_d, regblock_rhs, cacheBits
  );
};

/* CT = A * BT using cache blocking and 2x1x2 register blocking where possible.
   For internal use.
*/
static void gemmBinary_neon_L1_tile4x2x2(
  uint64_t * A, uint64_t * BT, int32_t * CT, int32_t alpha,
  uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t rowsA_orig, uint64_t rowsBT_orig,
  uint64_t lhsBlock, uint64_t rhsBlock) {
  const uint64_t Atile = 4, DepthTile = 2, BTtile = 2;
  assert(rowsBT % rhsBlock == 0);
  assert(rowsA % lhsBlock == 0);
  assert(lhsBlock % Atile == 0);
  assert(rhsBlock % BTtile == 0);

  for(uint64_t bBT = 0; bBT < rowsBT; bBT += rhsBlock) {
    for(uint64_t bA = 0; bA < rowsA; bA += lhsBlock) {
      const size_t num_acc = Atile*BTtile;
      // start of cache block
      for(uint64_t rBT = bBT; rBT < bBT + rhsBlock; rBT += BTtile) {
        uint64_t * BTptr = &BT[rBT * depth_words];
        for(uint64_t rA = bA; rA < bA + lhsBlock; rA += Atile) {
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
              if(((rBT + bt) < rowsBT_orig) && ((rA + at) < rowsA_orig)) {
                CT[(rBT + bt) * rowsA_orig + (rA + at)] += acc[at * BTtile + bt] * alpha;
              }
            }
          }
        }
      }
      // end of cache block
    }
  }
}

/* Bit-serial GEMM via a series of calls to gemmBinary.
   Note that rhs must be given in transposed form, and the result is also
   produced transposed.
*/
static void gemmBitSerial_neon_usingBinary(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  const uint64_t lhsbits = ctx.lhs.nbits;
  const uint64_t rhsbits = ctx.rhs.nbits;
  // clear contents of result matrix by setting everything to zero
  memset(ctx.res, 0, ctx.lhs.nrows*ctx.rhs.nrows*sizeof(int32_t));
  // call binary GEMM for each bit position
  for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
    bool neg_lhs = ctx.lhs.issigned && (lbit == lhsbits-1);
    for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
      bool neg_rhs = ctx.rhs.issigned && (rbit == rhsbits-1);
      bool neg = neg_rhs ^ neg_lhs;
      int32_t alpha = neg ? -(1 << (lbit+rbit)) : (1 << (lbit+rbit));
      gemmBinary_neon_L1_tile4x2x2(
        ctx.lhs.bitplaneptr(lbit), ctx.rhs.bitplaneptr(rbit), ctx.res, alpha,
        ctx.lhs.nrows_a, ctx.lhs.wordsPerRow(), ctx.rhs.nrows_a,
        ctx.lhs.nrows, ctx.rhs.nrows, ctx.lhsBlock, ctx.rhsBlock
      );
    }
  }
}
