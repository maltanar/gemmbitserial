#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"

using namespace std;

int main(int argc, char const *argv[]) {
  tellmeall();
  size_t b = 2;
  size_t d = 1024;
  uint8_t * rnd_mat = new uint8_t[d*d];
  uint8_t * rnd_vec = new uint8_t[d];
  ResultVector res_golden;
  generateRandomVector(b, d, rnd_vec);
  generateRandomVector(b, d*d, rnd_mat);
  BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
  ResultVector resvec;
  auto start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < 100; i++)
    resvec = bitSerialMatrixVector(bsm, bsv, d);
  auto end = chrono::high_resolution_clock::now();
  float mscount = chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 100.0;

  cout << "Time: " << mscount << " ms" << endl;

  delete [] rnd_mat;
  delete [] rnd_vec;
  return 0;
}
