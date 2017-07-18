#pragma once
// generic (non-architecture-specific) implementations of gemmBitserial
// and other related functions

// Compute the row-wise sum of a bit-serial matrix
static void sumRows_generic(BitSerialMatrix m, int32_t * row_sums) {
  const uint64_t nc = m.wordsPerRow();

  for(uint64_t r = 0; r < m.nrows; r++) {
    int32_t row_acc = 0;
    if(m.isBipolar()) {
      uint64_t * rowptr = m.rowptr(0, r);
      for(uint64_t c = 0; c < nc; c++) {
        row_acc += __builtin_popcountll(rowptr[c]);
      }
      // account for -1s in the sum. how does this work? let p be the number of
      // +1 bits, and n be the number of -1 bits. we know that there are only
      // p+n bits in total, and we want to compute the sum as p-n. rewriting
      // -n in terms of the number of columns (bits), we get:
      row_sums[r] = 2 * row_acc - m.ncols;
    } else {
      for(uint64_t b = 0; b < m.nbits; b++) {
        uint64_t * rowptr = m.rowptr(b, r);
        int32_t bit_acc = 0;
        for(uint64_t c = 0; c < nc; c++) {
          bit_acc += __builtin_popcountll(rowptr[c]);
        }
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

static void prepareAccumulators_generic(GEMMContext ctx) {
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
    sumRows_generic(regularM, rowwise_sum);
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

static GEMMContext allocGEMMContext_generic(
  uint64_t lhsRows, uint64_t depth, uint64_t rhsRows,
  uint64_t lhsBits, uint64_t rhsBits,
  bool lhsSigned, bool rhsSigned
) {
  const uint64_t regblock_lhs = 2;
  const uint64_t regblock_d = 1;
  const uint64_t regblock_rhs = 2;
  const uint64_t cacheBits = 32*1024*8;

  if(rhsRows == 1) {
    // matrix-vector only needs depth alignment
    return allocGEMMContext_base(
      lhsRows, depth, rhsRows, lhsBits, rhsBits, lhsSigned, rhsSigned,
      1, 4, 1, cacheBits
    );
  } else {
    return allocGEMMContext_base(
      lhsRows, depth, rhsRows, lhsBits, rhsBits, lhsSigned, rhsSigned,
      regblock_lhs, regblock_d, regblock_rhs, cacheBits
    );
  }
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
static void gemmBinary_generic_L1_tile2x1x2(
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
static void gemmBitSerial_generic_usingBinary(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  const uint64_t lhsbits = ctx.lhs.nbits;
  const uint64_t rhsbits = ctx.rhs.nbits;

  prepareAccumulators_generic(ctx);
  // call binary GEMM for each bit position
  // note that bipolars don't count as negative, we do those with {0, 1} as a
  // special case
  for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
    bool neg_lhs = ctx.lhs.issigned && !ctx.lhs.isBipolar() && (lbit == lhsbits-1);
    for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
      bool neg_rhs = ctx.rhs.issigned && !ctx.rhs.isBipolar() && (rbit == rhsbits-1);
      bool neg = neg_rhs ^ neg_lhs;
      int32_t alpha = neg ? -(1 << (lbit+rbit)) : (1 << (lbit+rbit));
      alpha = ctx.isBipolarTimesRegular() ? 2*alpha : alpha;
      gemmBinary_generic_L1_tile2x1x2(
        ctx.lhs.bitplaneptr(lbit), ctx.rhs.bitplaneptr(rbit), ctx.res, alpha,
        ctx.lhs.nrows_a, ctx.lhs.wordsPerRow(), ctx.rhs.nrows_a,
        ctx.lhs.nrows, ctx.rhs.nrows, ctx.lhsBlock, ctx.rhsBlock
      );
    }
  }
}


/* Standalone bit-serial GEMM. Note that rhs must be given in transposed
   form, and the result is also produced transposed.
*/
static void gemmBitSerial_generic_naive(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  const uint64_t lhsbits = ctx.lhs.nbits;
  const uint64_t rhsbits = ctx.rhs.nbits;
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t out_cols = ctx.rhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  prepareAccumulators_generic(ctx);
  for(uint64_t i = 0; i < out_cols; i++) {
    for(uint64_t j = 0; j < out_rows; j++) {
      int32_t rowres = 0;
      for(uint64_t lbit = 0; lbit < lhsbits; lbit++) {
        bool neg_lhs = ctx.lhs.issigned && !ctx.lhs.isBipolar() && (lbit == lhsbits-1);
        for(uint64_t rbit = 0; rbit < rhsbits; rbit++) {
          bool neg_rhs = ctx.rhs.issigned && !ctx.rhs.isBipolar() && (rbit == rhsbits-1);
          uint64_t * ldata = ctx.lhs.rowptr(lbit, j);
          uint64_t * rdata = ctx.rhs.rowptr(rbit, i);
          uint64_t andcard = 0;
          // AND-popcount-accumulate over row pair
          for(uint64_t k = 0; k < depth; k++) {
            andcard += __builtin_popcountll(ldata[k] & rdata[k]);
          }
          // scale
          uint64_t bpreg_scale = ctx.isBipolarTimesRegular() ? 1 : 0;
          andcard = andcard << (lbit + rbit + bpreg_scale);
          // negate if needed
          rowres += (neg_lhs ^ neg_rhs) ? -andcard : andcard;
        }
      }
      ctx.res[i * ctx.lhs.nrows + j] += rowres;
    }
  }
}

// Special case: bipolar times bipolar matrix multiplication. These use
// XNOR-popcount instead of AND-popcount, and also need an additional correction
// step to account for zeroes being treated as -1 bits

// naive implementation for bipolar GEMM
static void gemmBipolar_generic_naive(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  assert(ctx.lhs.isBipolar() && ctx.rhs.isBipolar());
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t out_cols = ctx.rhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  prepareAccumulators_generic(ctx);
  for(uint64_t i = 0; i < out_cols; i++) {
    for(uint64_t j = 0; j < out_rows; j++) {
      int32_t rowres = 0;
      uint64_t * ldata = ctx.lhs.rowptr(0, j);
      uint64_t * rdata = ctx.rhs.rowptr(0, i);
      // XNOR-popcount-accumulate over row pair. note that we do XOR-popcount
      // to save one instruction (no need to invert the XOR result). this is
      // accounted for in the correction afterwards.
      for(uint64_t k = 0; k < depth; k++) {
        rowres += __builtin_popcountll(ldata[k] ^ rdata[k]);
      }
      // correction for sum of 1 and -1 bits
      ctx.res[i * ctx.lhs.nrows + j] +=  -2 * rowres + ctx.lhs.ncols;
    }
  }
}

/* Standalone bit-serial GEMV (matrix-vector). Note that rhs must be given in transposed
   form, and the result is also produced transposed.
*/
static void gemvBitSerial_generic(GEMMContext ctx) {
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
        uint64_t andcard = 0;
        // AND-popcount-accumulate over row pair
        for(uint64_t k = 0; k < depth; k+=4) {
          andcard += __builtin_popcountll(ldata[k] & rdata[k]);
          andcard += __builtin_popcountll(ldata[k+1] & rdata[k+1]);
          andcard += __builtin_popcountll(ldata[k+2] & rdata[k+2]);
          andcard += __builtin_popcountll(ldata[k+3] & rdata[k+3]);
        }
        // scale
        andcard = andcard << (lbit + rbit + bpreg_scale);
        // negate if needed
        rowres += (neg_lhs ^ neg_rhs) ? -andcard : andcard;
      }
    }
    ctx.res[j] += rowres;
  }
}

// Special case: bipolar times bipolar matrix vector multiplication. These use
// XNOR-popcount instead of AND-popcount, and also need an additional correction
// step to account for zeroes being treated as -1 bits

static void gemvBipolar_generic(GEMMContext ctx) {
  // ensure that matrix shapes are compatible
  assert(ctx.lhs.ncols == ctx.rhs.ncols);
  assert(ctx.lhs.isBipolar() && ctx.rhs.isBipolar());
  const uint64_t out_rows = ctx.lhs.nrows;
  const uint64_t depth = ctx.lhs.wordsPerRow();
  prepareAccumulators_generic(ctx);
  for(uint64_t j = 0; j < out_rows; j++) {
    int32_t rowres = 0;
    uint64_t * ldata = ctx.lhs.rowptr(0, j);
    uint64_t * rdata = ctx.rhs.rowptr(0, 0);
    // XNOR-popcount-accumulate over row pair. note that we do XOR-popcount
    // to save one instruction (no need to invert the XOR result). this is
    // accounted for in the correction afterwards.
    for(uint64_t k = 0; k < depth; k+=4) {
      rowres += __builtin_popcountll(ldata[k] ^ rdata[k]);
      rowres += __builtin_popcountll(ldata[k+1] ^ rdata[k+1]);
      rowres += __builtin_popcountll(ldata[k+2] ^ rdata[k+2]);
      rowres += __builtin_popcountll(ldata[k+3] ^ rdata[k+3]);
    }
    // correction for sum of 1 and -1 bits
    ctx.res[j] +=  -2 * rowres + ctx.lhs.ncols;
  }
}

static void gemmBitSerial_generic(GEMMContext ctx) {
  if(ctx.isMatrixVector()) {
    if(ctx.isBipolarTimesBipolar()) {
      gemvBipolar_generic(ctx);
    } else {
      gemvBitSerial_generic(ctx);
    }
  } else {
    if(ctx.isBipolarTimesBipolar()) {
      gemmBipolar_generic_naive(ctx);
    } else {
      gemmBitSerial_generic_usingBinary(ctx);
    }
  }
}
