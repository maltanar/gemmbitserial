#!/bin/bash

g++ -std=c++11 -O3 -march=native *.cpp -fPIC -c
ar -cvq libgemmbitserial.a *.o

