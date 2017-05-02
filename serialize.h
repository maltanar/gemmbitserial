#pragma once
#include "convert.h"
#include <iostream>

/**
* Print contents of different vector types
*/
std::ostream &operator<<(std::ostream &os, BitVector const &m);
std::ostream &operator<<(std::ostream &os, BitSerialVector const &m);
std::ostream &operator<<(std::ostream &os, BitSerialMatrix const &m);
std::ostream &operator<<(std::ostream &os, ResultVector const &m);
std::ostream &operator<<(std::ostream &os, AccumulateVector const &m);

/**
* de/serialize gemm-bitserial types from/to binary streams
*/
void serialize(std::ostream &os, MyBitVector const &m);
void serialize(std::ostream &os, BitSerialVector const &m);
void serialize(std::ostream &os, BitSerialMatrix const &m);

MyBitVector deserializeMyBitVector(std::istream &is);
BitSerialVector deserializeBitSerialVector(std::istream &is);
BitSerialMatrix deserializeBitSerialMatrix(std::istream &is);
