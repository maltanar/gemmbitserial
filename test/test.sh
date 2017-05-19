#!/bin/bash

if [ -f test.out ]; then
    rm test.out
fi
g++ -march=native -g -std=c++11 -I.. test.cpp -o test.out
./test.out
