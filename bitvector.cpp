#include "bitvector.h"
#include <cmath>

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
  const uint64_t *bufptrA = m_buf.data();
  const uint64_t *bufptrB = rhs.m_buf.data();
  for(size_t i = 0; i < m_numWords; i++) {
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
