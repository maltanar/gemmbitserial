#pragma once
#include <stdint.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <math.h>

namespace gemmbitserial {

// Utility function to increment-and-align "in" to "af"
inline uint64_t alignTo(uint64_t in, uint64_t af) {
  return in + (af - (in % af));
}

class BitSerialMatrix {
public:
  // static member functions for working with BitSerialMatrix

  /* Allocate buffer space for a BitSerialMatrix */
  static BitSerialMatrix alloc(uint64_t nbits, uint64_t nrows, uint64_t ncols, bool issigned, uint64_t rowalign = 1, uint64_t colalign = 64) {
    BitSerialMatrix bsm;
    bsm.nbits = nbits;
    bsm.nrows = nrows;
    bsm.ncols = ncols;
    bsm.nrows_a = alignTo(nrows, rowalign);
    bsm.ncols_a = alignTo(ncols, colalign);
    bsm.issigned = issigned;
    uint64_t wordsPerBitplane = bsm.wordsPerBitplane();
    bsm.data = new uint64_t[nbits * wordsPerBitplane];
    return bsm;
  }

  /* Deallocate buffers for a BitSerialMatrix */
  static void dealloc(BitSerialMatrix bsm) {
    delete [] bsm.data;
  }

public:
  // actual member variables and functions of BitSerialMatrix instances
  bool issigned;        // whether highest order bit pos is negative
  uint64_t nbits;       // bits of precision
  uint64_t nrows;       // number of real (actual) rows
  uint64_t ncols;       // number of real (actual) columns
  uint64_t nrows_a;     // number of allocated rows
  uint64_t ncols_a;     // number of allocated columns
  uint64_t * data;      // data buffer, layout [nbits][nrows_a][ncols_a/64]

  // number of storage words needed for each row
  inline uint64_t wordsPerRow() const {
    const uint64_t bitsPerWord = sizeof(uint64_t) * 8;
    return ncols_a / bitsPerWord;
  }

  // number of storage words needed for each bitplane (bit matrix)
  inline uint64_t wordsPerBitplane() const {
    return nrows_a * wordsPerRow();
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

  /* Imports a regular matrix into this BitSerialMatrix.
  */
  template <typename T>
  void importRegular(T * matrix, bool readColMajor=false) {
    // TODO add support for transposed reading
    assert(!readColMajor);
    this->clearAll();
    for(uint64_t r = 0; r < this->nrows; r++) {
      for(uint64_t c = 0; c < this->ncols; c++) {
        uint8_t currentElem = (uint8_t) matrix[r * this->ncols + c];
        for(uint64_t b = 0; b < this->nbits; b++) {
          if(currentElem & (1 << b)) {
            this->set(b, r, c);
          }
        }
      }
    }
  }

  /* Convert this BitSerialMatrix back to a regular matrix.
  */
  template <typename T>
  void exportRegular(T * matrix) {
    for(uint64_t r = 0; r < this->nrows; r++) {
      for(uint64_t c = 0; c < this->ncols; c++) {
        uint8_t currentElem = 0;
        for(uint64_t b = 0; b < this->nbits; b++) {
          if(this->get(b, r, c)) {
            currentElem |= 1 << b;
          }
        }
        matrix[r * this->ncols + c] = (T) currentElem;
      }
    }
  }
};

/* Utility function to find block size under the following assumptions:
   - size of lhs block + rhs block + result block <= cacheBits
   - no blocking along depth (i.e. only entire rows of dBits bits)
   - lhsMult and rhsMult determine the ratio for lhs and rhs rows in cache
   - returned lhsRows and rhsRows are divisible by lhsMult and rhsMult, respectively
   - each result elem takes bitsPerRes bits
*/
void computeBlockSize(float lhsMult, float rhsMult, float cacheBits, float dBits, uint64_t & lhsBlock, uint64_t & rhsBlock) {
  float a = sizeof(int32_t) * lhsMult * rhsMult;
  float b = dBits*(lhsMult + rhsMult);
  float c = -cacheBits;
  float discr = sqrt(b*b - 4 * a * c);
  assert(discr > 0);
  int64_t x0 = floor((-b + discr) / (2*a));
  int64_t x1 = floor((-b - discr) / (2*a));
  int64_t x = x0 > x1 ? x0 : x1;
  assert(x > 0);
  lhsBlock = lhsMult * x;
  rhsBlock = rhsMult * x;
};


class GEMMContext {
public:
  BitSerialMatrix lhs, rhs;
  uint64_t lhsBlock, rhsBlock;
  int32_t * res;
};

void deallocGEMMContext(GEMMContext ctx) {
  delete [] ctx.res;
  BitSerialMatrix::dealloc(ctx.lhs);
  BitSerialMatrix::dealloc(ctx.rhs);
};

// generic implementations using regular & and __builtin_popcountll
#include "arch-generic.hpp"

// select the implementations to be used based on architecture
#if defined(__ARM_NEON) || defined(__aarch64__)
#warning "Compiling with ARM NEON"
#include "arch-neon.hpp"
// ARM NEON-specific implementations
#define gemmBitSerial   gemmBitSerial_neon_usingBinary
// TODO context def
#else
#warning "Compiling using generic popcount"
#define gemmBitSerial     gemmBitSerial_generic_usingBinary
#define allocGEMMContext  allocGEMMContext_generic
#endif

}
