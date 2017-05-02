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
