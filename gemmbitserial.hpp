#pragma once
#include <stdint.h>
#include <string.h>
#include <cassert>
#include <iostream>

namespace gemmbitserial {

class BitSerialMatrix {
public:
  bool issigned;        // whether highest order bit pos is negative
  uint64_t nbits;       // bits of precision
  uint64_t nrows;       // number of rows
  uint64_t ncols;       // number of columns
  uint64_t * data;    // data buffer, layout [nbits][nrows][ceil(ncols/64)]

  // number of storage words needed for each row
  inline uint64_t wordsPerRow() const {
    const uint64_t bitsPerWord = sizeof(uint64_t) * 8;
    return (ncols + bitsPerWord - 1) / bitsPerWord;
  }

  // number of storage words needed for each bitplane (bit matrix)
  inline uint64_t wordsPerBitplane() const {
    return nrows * wordsPerRow();
  }

  // get given bit. true if set, false if unset.
  inline bool get(uint64_t bit, uint64_t row, uint64_t col) {
    return ((word(bit, row, col) >> bitpos(col)) & 1L) == 1;
  }

  // set all bits to zero
  inline void clearAll() {
    memset(data, 0, nbits * wordsPerBitplane() * sizeof(uint64_t));
  }

  // set given bit to one
  inline void set(uint64_t bit, uint64_t row, uint64_t col) {
    word(bit, row, col) |= (1L << bitpos(col));
  }

  // set given bit to zero
  inline void unset(uint64_t bit, uint64_t row, uint64_t col) {
    word(bit, row, col) &= ~(1L << bitpos(col));
  }

  // access the container word for a given bit
  inline uint64_t & word(uint64_t bit, uint64_t row, uint64_t col) {
    // right shift by log2(bits per word) to get word index
    uint64_t colw = col >> 6;
    return data[bit * wordsPerBitplane() + row * wordsPerRow() + colw];
  }

  // get a pointer to a particular row
  inline uint64_t * rowptr(uint64_t bit, uint64_t row) {
    return &data[bit * wordsPerBitplane() + row * wordsPerRow()];
  }

  // get a pointer to a particular bit plane
  inline uint64_t * bitplaneptr(uint64_t bit) {
    return &data[bit * wordsPerBitplane()];
  }

  uint64_t bitpos(uint64_t col) {
    // return modulo 64 of col by using a bitmask
    return col & ((1 << 6) - 1);
  }
};

/* Allocate buffer space for a BitSerialMatrix */
void allocBitSerialMatrix(BitSerialMatrix * bsm, uint64_t nbits, uint64_t nrows, uint64_t ncols, bool issigned) {
  bsm->nbits = nbits;
  bsm->nrows = nrows;
  bsm->ncols = ncols;
  bsm->issigned = issigned;
  uint64_t wordsPerBitplane = bsm->wordsPerBitplane();
  bsm->data = new uint64_t[nbits * wordsPerBitplane];
}

/* Deallocate buffers for a BitSerialMatrix */
void deallocBitSerialMatrix(BitSerialMatrix * bsm) {
  delete [] bsm->data;
}

/* Convert given matrix to bit serial form. The BitSerialMatrix must have been
   already allocated. The original matrix is assumed to be stored in row-major
   order unless readColMajor is specified.
*/
template <typename T>
void toBitSerialMatrix(T * matrix, BitSerialMatrix * bsm, bool readColMajor=false) {
  // TODO add support for transposed reading
  assert(!readColMajor);
  // clear buffer
  bsm->clearAll();

  for(uint64_t r = 0; r < bsm->nrows; r++) {
    for(uint64_t c = 0; c < bsm->ncols; c++) {
      uint8_t currentElem = (uint8_t) matrix[r * bsm->ncols + c];
      for(uint64_t b = 0; b < bsm->nbits; b++) {
        if(currentElem & (1 << b)) {
          bsm->set(b, r, c);
        }
      }
    }
  }
}

/* Convert a BitSerialMatrix back to a regular matrix.
*/
template <typename T>
void fromBitSerialMatrix(BitSerialMatrix * bsm, T * matrix) {
  for(uint64_t r = 0; r < bsm->nrows; r++) {
    for(uint64_t c = 0; c < bsm->ncols; c++) {
      uint8_t currentElem = 0;
      for(uint64_t b = 0; b < bsm->nbits; b++) {
        if(bsm->get(b, r, c)) {
          currentElem |= 1 << b;
        }
      }
      matrix[r * bsm->ncols + c] = (T) currentElem;
    }
  }
}

/* Multiply two binary matrices. Note that rhs must be given in transposed
   form, and the result is also produced transposed.
   TODO should we use standard alpha-beta as in standard GEMM here?
*/
template <typename AccType>
void gemmNaiveBinary(uint64_t * A, uint64_t * BT, AccType * CT, int leftshift,
uint64_t rowsA, uint64_t depth_words, uint64_t rowsBT, bool negate) {
  for(uint64_t rBT = 0; rBT < rowsBT; rBT++) {
    uint64_t * BTptr = &BT[rBT * depth_words];
    for(uint64_t rA = 0; rA < rowsA; rA++) {
      uint64_t * Aptr = &A[rA * depth_words];
      AccType acc = 0;
      for(uint64_t d = 0; d < depth_words; d++) {
        acc += __builtin_popcountll(Aptr[d] & BTptr[d]);
      }
      CT[rBT * rowsA + rA] += (acc << leftshift) * (negate ? -1 : +1);
    }
  }
}

template <typename AccType>
void gemmBitSerial_usingNaiveBinary(BitSerialMatrix * lhs, BitSerialMatrix * rhs, AccType * res) {
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
      gemmNaiveBinary(lhs->bitplaneptr(lbit), rhs->bitplaneptr(rbit), res, lbit+rbit, lhs->nrows, lhs->wordsPerRow(), rhs->nrows, neg);
    }
  }
}


/* Multiply two bit serial matrices. Note that rhs must be given in transposed
   form, and the result is also produced transposed.
*/
template <typename AccType>
void gemmBitSerial_naive(BitSerialMatrix * lhs, BitSerialMatrix * rhs, AccType * res) {
  // ensure that matrix shapes are compatible
  assert(lhs->ncols == rhs->ncols);
  const uint64_t lhsbits = lhs->nbits;
  const uint64_t rhsbits = rhs->nbits;
  const uint64_t out_rows = lhs->nrows;
  const uint64_t out_cols = rhs->nrows;
  const uint64_t depth = lhs->wordsPerRow();

  /* TODO add register blocking */
  /* TODO add cache blocking */
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
      // TODO increment or set here?
      res[i * out_rows + j] = rowres;
    }
  }
}

#define gemmBitSerial gemmBitSerial_usingNaiveBinary


}
