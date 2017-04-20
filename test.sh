#!/bin/bash

if [ -f test.out ]; then
    rm test.out
fi
g++ -march=native -O3 -std=c++11 *.cpp *.c -o test.out
./test.out
