#include "serialize.h"

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
