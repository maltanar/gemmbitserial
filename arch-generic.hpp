#pragma once

// generic (non-architecture-specific) implementations of gemmBitserial
// and other related functions

GEMMContext allocGEMMContext_generic(
  uint64_t lhsRows, uint64_t depth, uint64_t rhsRows,
  uint64_t lhsBits, uint64_t rhsBits,
  bool lhsSigned, bool rhsSigned
) {
  const uint64_t regblock_lhs = 2;
  const uint64_t regblock_d = 1;
  const uint64_t regblock_rhs = 2;
  const uint64_t cacheBits = 32*1024*8;

  GEMMContext ret;
  uint64_t depth_al = alignTo(depth, regblock_d*64);
  // TODO add bypass for small enough matrices here?
  // use cache blocking; compute sizes
  computeBlockSize(
    regblock_lhs, regblock_rhs, cacheBits, depth_al,
    ret.lhsBlock, ret.rhsBlock
  );
  if(ret.lhsBlock > lhsRows || ret.rhsBlock > rhsRows) {
    // use register blocking only
    ret.lhsBlock = alignTo(lhsRows, regblock_lhs);
    ret.rhsBlock = alignTo(rhsRows, regblock_rhs);
  }
  // allocate aligned bit serial matrices
  ret.lhs = BitSerialMatrix::alloc(
    lhsBits, lhsRows, depth, lhsSigned, ret.lhsBlock, regblock_d*64
  );
  ret.rhs = BitSerialMatrix::alloc(
    rhsBits, rhsRows, depth, rhsSigned, ret.rhsBlock, regblock_d*64
  );
  // allocate result matrix. note that it is not aligned -- the
  // elements corresponding to alignment parts won't materialize.
  ret.res = new int32_t[lhsRows * rhsRows];
  return ret;
};



/* Multiply a lhs_block x rhs_block chunk of the given matrices, starting at
  (bA, bBT) using 2x1x2 register tiling. For internal use.
*/
inline void gemmBinary_generic_chunk_tile2x1x2(
  uint64_t * A, uint64_t * BT, int32_t * CT,
  int32_t alpha,
  uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t bA, uint64_t bBT,
  uint64_t lhs_block, uint64_t rhs_block,
  uint64_t rowsA_orig, uint64_t rowsBT_orig) {
  const uint64_t Atile = 2, DepthTile = 1, BTtile = 2;
  const size_t num_acc = Atile*BTtile;

  for(uint64_t rBT = bBT; rBT < bBT + rhs_block; rBT += BTtile) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = bA; rA < bA + lhs_block; rA += Atile) {
      uint64_t * Aptr = &A[rA * depth_words];
      int32_t acc[num_acc] = {0};
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
          if(((rBT + bt) < rowsBT_orig) && ((rA + at) < rowsA_orig)) {
            CT[(rBT + bt) * rowsA_orig + (rA + at)] += acc[at * BTtile + bt] * alpha;
          }
        }
      }
    }
  }
}

/* CT = A * BT using cache blocking and 2x1x2 register blocking where possible.
   For internal use.
*/
void gemmBinary_generic_L1_tile2x1x2(
  uint64_t * A, uint64_t * BT, int32_t * CT, int32_t alpha,
  uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT,
  uint64_t rowsA_orig, uint64_t rowsBT_orig,
  uint64_t lhsBlock, uint64_t rhsBlock
  ) {
  const uint64_t Atile = 2, DepthTile = 1, BTtile = 2;
  assert(rowsBT % rhsBlock == 0);
  assert(rowsA % lhsBlock == 0);
  assert(lhsBlock % Atile == 0);
  assert(rhsBlock % BTtile == 0);

  for(uint64_t bBT = 0; bBT < rowsBT; bBT += rhsBlock) {
    for(uint64_t bA = 0; bA < rowsA; bA += lhsBlock) {
      gemmBinary_generic_chunk_tile2x1x2(
        A, BT, CT, alpha, rowsA, depth_words, rowsBT, bA, bBT,
        lhsBlock, rhsBlock, rowsA_orig, rowsBT_orig
      );
    }
  }
}

/* Bit-serial GEMM via a series of calls to gemmBinary.
   Note that rhs must be given in transposed form, and the result is also
   produced transposed.
*/

void gemmBitSerial_generic_usingBinary(GEMMContext ctx) {
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
      gemmBinary_generic_L1_tile2x1x2(
        ctx.lhs.bitplaneptr(lbit), ctx.rhs.bitplaneptr(rbit), ctx.res, alpha,
        ctx.lhs.nrows_a, ctx.lhs.wordsPerRow(), ctx.rhs.nrows_a,
        ctx.lhs.nrows, ctx.rhs.nrows, ctx.lhsBlock, ctx.rhsBlock
      );
    }
  }
}
