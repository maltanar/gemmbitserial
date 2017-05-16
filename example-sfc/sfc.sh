#!/bin/bash

if [ -f sfc.out ]; then
    rm sfc.out
fi
g++ -march=native -O3 -std=c++11 -I.. sfc.cpp ../*.cpp -o sfc.out
./sfc.out
