#pragma once
#include <vector>
#include <stdint.h>
#include <string.h>

class MyBitVector {
public:
  MyBitVector(size_t numBits);

  void add(uint64_t index);
  void clear();
  bool contains(uint64_t index) const;
  uint64_t and_cardinality(const MyBitVector & rhs) const;

protected:
  std::vector<uint64_t> m_buf;
  size_t m_numWords, m_bufWordBits;
  uint64_t m_posShift, m_posMask;

  uint64_t getWPos(uint64_t bitIndex) const;
  uint64_t getBPos(uint64_t bitIndex) const;
};
