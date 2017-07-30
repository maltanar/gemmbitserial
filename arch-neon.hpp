#pragma once

// architecture-specific implementations of primitives for ARM NEON
// the tiling strategies are named as:
// _tile<lhs-rows><depth-elems><rhs-rows>

static inline uint64_t popcount_neon(uint64_t * rowptr, uint64_t numElems) {
  uint64_t ret = 0;
  const uint64_t DepthTile = 2;
  uint8x16_t acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
  uint64x2_t acc2_neon = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
  for(uint64_t c = 0; c < numElems; c += DepthTile) {
    uint8x16_t a0 = vld1q_u8((uint8_t *) &rowptr[c]);
    acc_neon = vaddq_u8(acc_neon, vcntq_u8(a0));
    if((c & 7L) == 7L) {
      // hsum over 8-bit accumulators when end or overflow
      acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
      acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    }
  }
  // move into regular accumulators
  uint64_t tmp[2];
  acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
  vst1q_u64(tmp, acc2_neon);
  ret = (tmp[0] + tmp[1]);
  return ret;
}

static inline uint64_t xor_popcount_neon(uint64_t * rowptrA, uint64_t * rowptrB, uint64_t numElems) {
  uint64_t ret = 0;
  const uint64_t DepthTile = 2;
  uint8x16_t acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
  uint64x2_t acc2_neon = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
  for(uint64_t c = 0; c < numElems; c += DepthTile) {
    uint8x16_t a0 = vld1q_u8((uint8_t *) &rowptrA[c]);
    uint8x16_t b0 = vld1q_u8((uint8_t *) &rowptrB[c]);
    acc_neon = vaddq_u8(acc_neon, vcntq_u8(veorq_u8(a0, b0)));
    if((c & 7L) == 7L) {
      // hsum over 8-bit accumulators when end or overflow
      acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
      acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    }
  }
  // move into regular accumulators
  uint64_t tmp[2];
  acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
  vst1q_u64(tmp, acc2_neon);
  ret = (tmp[0] + tmp[1]);
  return ret;
}

static inline uint64_t and_popcount_neon(uint64_t * rowptrA, uint64_t * rowptrB, uint64_t numElems) {
  uint64_t ret = 0;
  const uint64_t DepthTile = 2;
  uint8x16_t acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
  uint64x2_t acc2_neon = vcombine_u64(vcreate_u64(0), vcreate_u64(0));
  for(uint64_t c = 0; c < numElems; c += DepthTile) {
    uint8x16_t a0 = vld1q_u8((uint8_t *) &rowptrA[c]);
    uint8x16_t b0 = vld1q_u8((uint8_t *) &rowptrB[c]);
    acc_neon = vaddq_u8(acc_neon, vcntq_u8(vandq_u8(a0, b0)));
    if((c & 7L) == 7L) {
      // hsum over 8-bit accumulators when end or overflow
      acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
      acc_neon = vcombine_u8(vcreate_u8(0), vcreate_u8(0));
    }
  }
  // move into regular accumulators
  uint64_t tmp[2];
  acc2_neon = vaddq_u64(acc2_neon, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(acc_neon))));
  vst1q_u64(tmp, acc2_neon);
  ret = (tmp[0] + tmp[1]);
  return ret;
}

// Compute the row-wise sum of a bit-serial matrix
static void sumRows_neon(BitSerialMatrix m, int32_t * row_sums) {
  const uint64_t nc = m.wordsPerRow();

  for(uint64_t r = 0; r < m.nrows; r++) {
    int32_t row_acc = 0;
    if(m.isBipolar()) {
      uint64_t * rowptr = m.rowptr(0, r);
      row_acc += popcount_neon(rowptr, nc);
      // account for -1s in the sum. how does this work? let p be the number of
      // +1 bits, and n be the number of -1 bits. we know that there are only
      // p+n bits in total, and we want to compute the sum as p-n. rewriting
      // -n in terms of the number of columns (bits), we get:
      row_sums[r] = 2 * row_acc - m.ncols;
    } else {
      for(uint64_t b = 0; b < m.nbits; b++) {
        uint64_t * rowptr = m.rowptr(b, r);
        int32_t bit_acc = (int32_t) popcount_neon(rowptr, nc);
        // scale for weight and handle sign bit
        bit_acc = bit_acc << b;
        if(m.issigned && b == m.nbits - 1) {
          bit_acc = -bit_acc;
        }
        row_acc += bit_acc;
      }
      row_sums[r] = row_acc;
    }
  }
}

static void prepareAccumulators_neon(GEMMContext ctx) {
  // when bits = 1 and signed = true, we assume a matrix is bipolar, not using
  //{-1, 0} but instead {-1, +1} values.
  bool lhsBipolar = (ctx.lhs.nbits == 1) && ctx.lhs.issigned;
  bool rhsBipolar = (ctx.rhs.nbits == 1) && ctx.rhs.issigned;

  if(lhsBipolar ^ rhsBipolar) {
    BitSerialMatrix bipolarM = lhsBipolar ? ctx.lhs : ctx.rhs;
    BitSerialMatrix regularM = lhsBipolar ? ctx.rhs : ctx.lhs;
    // if only one matrix is bipolar, we'll need to do something special.
    // despite the bipolar matrix, we'll compute the result using {0,1}
    // (regular unsigned 1-bit) matrices as follows:
    // let x be a column vector, W a bipolar matrix, and B a binary matrix which
    // is identical to W except all -1s are represented as 0.
    // note that each element We in W can be rewritten as 2*Be-1
    // by initializing the result vector to the negative of sum of all elements
    // in x, we get the same result using B instead of W.
    // compute columnwise sum of the regular matrix with bit serial
    // TODO should this buffer be part of the GEMMContext?
    int32_t * rowwise_sum = new int32_t[regularM.nrows];
    sumRows_neon(regularM, rowwise_sum);
    // initialize result matrix accumulators from sum
    for(auto res_row = 0; res_row < ctx.rhs.nrows; res_row++) {
      for(auto res_col = 0; res_col < ctx.lhs.nrows; res_col++) {
        if(lhsBipolar) {
          ctx.res[res_row * ctx.lhs.nrows + res_col] = -rowwise_sum[res_row];

        } else {
          ctx.res[res_row * ctx.lhs.nrows + res_col] = -rowwise_sum[res_col];
        }
      }
    }
    delete [] rowwise_sum;
  } else {
    // just initialize all result matrix accumulators to zero
    memset(ctx.res, 0, ctx.lhs.nrows*ctx.rhs.nrows*sizeof(int32_t));
  }
}

static GEMMContext allocGEMMContext_neon(
  uint64_t lhsRows, uint64_t depth, uint64_t rhsRows,
  uint64_t lhsBits, uint64_t rhsBits,
  bool lhsSigned, bool rhsSigned
) {
  const uint64_t regblock_lhs = 4;
  const uint64_t regblock_d = 2;
  const uint64_t regblock_rhs = 2;
  const uint64_t cacheBits = 16*1024*8;

  if(rhsRows == 1) {
    // matrix-vector only needs depth alignment
    return allocGEMMContext_base(
      lhsRows, depth, rhsRows, lhsBits, rhsBits, lhsSigned, rhsSigned,
      1, regblock_d, 1, cacheBits
    );
  } else {
    return allocGEMMContext_base(
      lhsRows, depth, rhsRows, lhsBits, rhsBits, lhsSigned, rhsSigned,
      regblock_lhs, regblock_d, regblock_rhs, cacheBits
    );
  }
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
  prepareAccumulators_neon(ctx);
  // call binary GEMM for each bit position
  for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
    bool neg_lhs = ctx.lhs.issigned && !ctx.lhs.isBipolar() && (lbit == lhsbits-1);
    for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
      bool neg_rhs = ctx.rhs.issigned && !ctx.rhs.isBipolar() && (rbit == rhsbits-1);
      bool neg = neg_rhs ^ neg_lhs;
      int32_t alpha = neg ? -(1 << (lbit+rbit)) : (1 << (lbit+rbit));
      alpha = ctx.isBipolarTimesRegular() ? 2*alpha : alpha;
      gemmBinary_neon_L1_tile4x2x2(
        ctx.lhs.bitplaneptr(lbit), ctx.rhs.bitplaneptr(rbit), ctx.res, alpha,
        ctx.lhs.nrows_a, ctx.lhs.wordsPerRow(), ctx.rhs.nrows_a,
        ctx.lhs.nrows, ctx.rhs.nrows, ctx.lhsBlock, ctx.rhsBlock
      );
    }
  }
}

// naive implementation for bipolar GEMM
static void gemmBipolar_neon_naive(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  assert(ctx.lhs.isBipolar() && ctx.rhs.isBipolar());
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t out_cols = ctx.rhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  prepareAccumulators_generic(ctx);
  for(uint64_t i = 0; i < out_cols; i++) {
    for(uint64_t j = 0; j < out_rows; j++) {
      uint64_t * ldata = ctx.lhs.rowptr(0, j);
      uint64_t * rdata = ctx.rhs.rowptr(0, i);
      // XNOR-popcount-accumulate over row pair. note that we do XOR-popcount
      // to save one instruction (no need to invert the XOR result). this is
      // accounted for in the correction afterwards.
      int32_t rowres = (int32_t) xor_popcount_neon(ldata, rdata, depth);
      // correction for sum of 1 and -1 bits
      ctx.res[i * ctx.lhs.nrows + j] +=  -2 * rowres + ctx.lhs.ncols;
    }
  }
}

// neon bipolar matrix times vector (GEMV)
static void gemvBipolar_neon(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  assert(ctx.lhs.isBipolar() && ctx.rhs.isBipolar());
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  prepareAccumulators_generic(ctx);
  for(uint64_t j = 0; j < out_rows; j++) {
    uint64_t * ldata = ctx.lhs.rowptr(0, j);
    uint64_t * rdata = ctx.rhs.rowptr(0, 0);
    // XNOR-popcount-accumulate over row pair. note that we do XOR-popcount
    // to save one instruction (no need to invert the XOR result). this is
    // accounted for in the correction afterwards.
    int32_t rowres = (int32_t) xor_popcount_neon(ldata, rdata, depth);
    // correction for sum of 1 and -1 bits
    ctx.res[j] +=  -2 * rowres + ctx.lhs.ncols;
  }
}

// neon bit serial matrix times vector (GEMV)
static void gemvBitSerial_neon(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  const uint64_t lhsbits = ctx.lhs.nbits;
  const uint64_t rhsbits = ctx.rhs.nbits;
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  uint64_t bpreg_scale = ctx.isBipolarTimesRegular() ? 1 : 0;
  prepareAccumulators_generic(ctx);
  for(uint64_t j = 0; j < out_rows; j++) {
    int32_t rowres = 0;
    for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
      bool neg_lhs = ctx.lhs.issigned && !ctx.lhs.isBipolar() && (lbit == lhsbits-1);
      for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
        bool neg_rhs = ctx.rhs.issigned && !ctx.rhs.isBipolar() && (rbit == rhsbits-1);
        uint64_t * ldata = ctx.lhs.rowptr(lbit, j);
        uint64_t * rdata = ctx.rhs.rowptr(rbit, 0);
        uint64_t andcard = (int32_t) and_popcount_neon(ldata, rdata, depth);
        // scale
        andcard = andcard << (lbit + rbit + bpreg_scale);
        // negate if needed
        rowres += (neg_lhs ^ neg_rhs) ? -andcard : andcard;
      }
    }
    ctx.res[j] += rowres;
  }
}

static void gemmBitSerial_neon(GEMMContext ctx) {
  if(ctx.isMatrixVector()) {
    if(ctx.isBipolarTimesBipolar()) {
      gemvBipolar_neon(ctx);
    } else {
      gemvBitSerial_neon(ctx);
    }
  } else {
    if(ctx.isBipolarTimesBipolar()) {
      gemmBipolar_neon_naive(ctx);
    } else {
      gemmBitSerial_neon_usingBinary(ctx);
    }
  }
}
