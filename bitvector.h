#pragma once
#include <vector>
#include <stdint.h>
#include <string.h>
#include <iostream>

class MyBitVector {
public:
  MyBitVector(size_t numBits);

  size_t size() const;
  void add(uint64_t index);
  void clear();
  bool contains(uint64_t index) const;
  uint64_t and_cardinality(const MyBitVector & rhs) const;

  friend void serialize(std::ostream &os, MyBitVector const &m);
  friend MyBitVector deserializeMyBitVector(std::istream &is);

protected:
  std::vector<uint64_t> m_buf;
  size_t m_numWords, m_bufWordBits, m_numBits;
  uint64_t m_posShift, m_posMask;

  uint64_t getWPos(uint64_t bitIndex) const;
  uint64_t getBPos(uint64_t bitIndex) const;
};
