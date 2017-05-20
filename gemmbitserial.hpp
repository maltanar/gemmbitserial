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

// select the implementations to be used based on architecture
#if defined(__ARM_NEON) || defined(__aarch64__)
#warning "Compiling with ARM NEON"
#include "arch-neon.hpp"
#define gemmBitSerial   gemmBitSerial_neon_usingBinary
#else// generic implementations using regular & and __builtin_popcountll
#include "arch-generic.hpp"
#warning "Compiling using generic popcount"
#define gemmBitSerial   gemmBitSerial_generic_usingBinary
#endif
// TODO

}
