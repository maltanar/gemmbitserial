#include "serialize.h"
#include <cassert>

std::ostream &operator<<(std::ostream &os, BitVector const &m) {
  os << "BitVector of " << m.size() << " bits : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << m.contains(i) << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, BitSerialVector const &m) {
  os << "BitSerialVector of " << m.size() << " bits of precision " << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << "Bit position " << i << ":" << std::endl;
    os << m[i] << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, BitSerialMatrix const &m) {
  os << "BitSerialMatrix of " << m.size() << " rows " << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << "Row " << i << ": " << m[i] << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, ResultVector const &m) {
  os << "ResultVector of " << m.size() << " elements : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << (int) m[i] << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, AccumulateVector const &m) {
  os << "AccumulateVector of " << m.size() << " elements : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << (int) m[i] << std::endl;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, FloatVector const &m) {
  os << "FloatVector of " << m.size() << " elements : {" << std::endl;
  for(size_t i = 0; i < m.size(); i++) {
    os << i << " = " << (float) m[i] << std::endl;
  }
  return os;
}

void serialize(std::ostream &os, MyBitVector const &m) {
  os.write((const char *) &(m.m_numBits), sizeof(size_t));
  os.write((const char *) &(m.m_bufWordBits), sizeof(size_t));
  os.write((const char *) &(m.m_numWords), sizeof(size_t));
  os.write((const char *) m.m_buf.data(), m.m_numWords * (m.m_bufWordBits/8) );
}

void serialize(std::ostream &os, BitSerialVector const &m) {
  size_t prec = m.size();
  os.write((const char *) &prec, sizeof(size_t));
  for(size_t i = 0; i < m.size(); i++) {
    serialize(os, m[i]);
  }
}

void serialize(std::ostream &os, BitSerialMatrix const &m) {
  size_t num_rows = m.size();
  os.write((const char *) &num_rows, sizeof(size_t));
  for(size_t i = 0; i < m.size(); i++) {
    serialize(os, m[i]);
  }
}

MyBitVector deserializeMyBitVector(std::istream &is) {
  size_t numBits, bufWordBits, numWords;
  is.read((char *) &numBits, sizeof(size_t));
  is.read((char *) &bufWordBits, sizeof(size_t));
  is.read((char *) &numWords, sizeof(size_t));
  MyBitVector ret(numBits);
  assert(ret.m_bufWordBits == bufWordBits);
  assert(ret.m_numWords == numWords);
  is.read((char *) ret.m_buf.data(), ret.m_numWords * (ret.m_bufWordBits/8));
  return ret;
}

BitSerialVector deserializeBitSerialVector(std::istream &is) {
  size_t prec;
  is.read((char *) &prec, sizeof(size_t));
  BitSerialVector ret;
  for(size_t i = 0; i < prec; i++) {
    ret.push_back(deserializeMyBitVector(is));
  }
  return ret;
}

BitSerialMatrix deserializeBitSerialMatrix(std::istream &is) {
  size_t num_rows;
  is.read((char *) &num_rows, sizeof(size_t));
  BitSerialMatrix ret;
  for(size_t i = 0; i < num_rows; i++) {
    ret.push_back(deserializeBitSerialVector(is));
  }
  return ret;
}
