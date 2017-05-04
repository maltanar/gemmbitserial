#include "bitvector.h"
#include <cmath>
#include <stdint.h>
#include <stddef.h>


MyBitVector::MyBitVector(size_t numBits) {
  m_bufWordBits = (sizeof(uint64_t) * 8);
  m_numWords = (numBits / m_bufWordBits) + 1;
  m_buf = std::vector<uint64_t>(m_numWords, 0);
  m_posShift = std::log2(m_bufWordBits);
  m_posMask = (1 << m_posShift) - 1;
  m_numBits = numBits;
}

size_t MyBitVector::size() const {
  return m_numBits;
}

void MyBitVector::clear() {
  for(auto &e: m_buf) {
    e = 0;
  }
}

void MyBitVector::add(uint64_t index) {
  const uint64_t w = m_buf[getWPos(index)];
  uint64_t wm = 1;
  wm = wm << getBPos(index);
  m_buf[getWPos(index)] = w | wm;
}

bool MyBitVector::contains(uint64_t index) const {
  bool ret = (m_buf[getWPos(index)] >> getBPos(index) & 1L) == 1;
  return ret;
}

uint64_t MyBitVector::and_cardinality(const MyBitVector & rhs) const {
  // TODO don't assume lengths are equal?
  uint64_t res = 0;
  const uint64_t * __restrict bufptrA = m_buf.data();
  const uint64_t * __restrict bufptrB = rhs.m_buf.data();
  const size_t rm = m_numWords % 4;
  const size_t rmd = m_numWords - rm;
  for(size_t i = 0; i < rmd; i+=4) {
    res += __builtin_popcountll(bufptrA[i] & bufptrB[i]);
    res += __builtin_popcountll(bufptrA[i+1] & bufptrB[i+1]);
    res += __builtin_popcountll(bufptrA[i+2] & bufptrB[i+2]);
    res += __builtin_popcountll(bufptrA[i+3] & bufptrB[i+3]);
  }
  for(size_t i = rmd; i < m_numWords; i++) {
    res += __builtin_popcountll(bufptrA[i] & bufptrB[i]);
  }

  return res;
}

inline uint64_t MyBitVector::getWPos(uint64_t bitIndex) const {
  return bitIndex >> m_posShift;
}

inline uint64_t MyBitVector::getBPos(uint64_t bitIndex) const {
  return bitIndex & m_posMask;
}
