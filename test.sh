#!/bin/bash

if [ -f test.out ]; then
    rm test.out
fi
g++ -march=native -O3 -std=c++11 test.cpp gemm-bitserial.cpp roaring.c -o test.out
./test.out
