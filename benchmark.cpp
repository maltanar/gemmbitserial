#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"

using namespace std;

int main(int argc, char const *argv[]) {
  size_t b = 2;
  size_t d = 1024;
  size_t reps = 100;
  uint8_t * rnd_mat = new uint8_t[d*d];
  uint8_t * rnd_vec = new uint8_t[d];
  ResultVector res_golden;
  generateRandomVector(b, d, rnd_vec);
  generateRandomVector(b, d*d, rnd_mat);
  BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
  ResultVector resvec;
  auto start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    resvec = bitSerialMatrixVector(bsm, bsv, d);
  auto end = chrono::high_resolution_clock::now();
  float uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  float perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector: " << uscount << " microseconds" << endl;
  cout << "Performance: " << perf << " binary ops per second" << endl;


  delete [] rnd_mat;
  delete [] rnd_vec;
  return 0;
}
