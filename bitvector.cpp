#include "bitvector.h"
#include <cmath>
#include <stdint.h>
#include <stddef.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

MyBitVector::MyBitVector(size_t numBits) {
  m_bufWordBits = (sizeof(uint64_t) * 8);
  m_numWords = (numBits / m_bufWordBits) + 1;
  // align to 64 bytes
  if(m_numWords % 8 != 0) {
    m_numWords += 8 - (m_numWords % 8);
  }
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
  uint64_t res = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
  const uint8_t * bufptrA = (const uint8_t *) m_buf.data();
  const uint8_t * bufptrB = (const uint8_t *) rhs.m_buf.data();
  uint64_t tmp[2];
  uint64_t chunk_size = 64;
  uint64_t n = (m_numWords*8) / chunk_size;

  uint8x16x4_t inputA0, inputB0;
  uint8x16x4_t inputA1, inputB1;
  uint8x16_t t0;
  uint32x4_t t1;

  uint64x2_t sum = vcombine_u64(vcreate_u64(0), vcreate_u64(0));

  for (uint64_t i = 0; i < n; i++, bufptrA += chunk_size, bufptrB += chunk_size)
  {
    inputA0 = vld4q_u8(bufptrA);
    inputB0 = vld4q_u8(bufptrB);

    t0 = vcntq_u8(vandq_u8(inputA0.val[0], inputB0.val[0]));
    t0 = vaddq_u8(t0, vcntq_u8(vandq_u8(inputA0.val[1], inputB0.val[1])));
    t0 = vaddq_u8(t0, vcntq_u8(vandq_u8(inputA0.val[2], inputB0.val[2])));
    t0 = vaddq_u8(t0, vcntq_u8(vandq_u8(inputA0.val[3], inputB0.val[3])));
    t1 = vpaddlq_u16(vpaddlq_u8(t0));

    sum = vpadalq_u32(sum, t1);
  }
  vst1q_u64(tmp, sum);
  for (uint64_t i = 0; i < 2; i++) {
    res += tmp[i];
  }
#else
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
#endif
  return res;
}

inline uint64_t MyBitVector::getWPos(uint64_t bitIndex) const {
  return bitIndex >> m_posShift;
}

inline uint64_t MyBitVector::getBPos(uint64_t bitIndex) const {
  return bitIndex & m_posMask;
}
