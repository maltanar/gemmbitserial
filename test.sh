#!/bin/sh
rm test.out
g++ -march=native -O3 -std=c++11 *.cpp *.c -o test.out
./test.out
