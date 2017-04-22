#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"

using namespace std;

int main(int argc, char const *argv[]) {
  size_t b = 2;
  size_t d = 1024;
  size_t thres_levels = 3;
  size_t reps = 100;
  uint8_t * rnd_mat = new uint8_t[d*d];
  uint8_t * rnd_vec = new uint8_t[d];
  generateRandomVector(b, d, rnd_vec);
  generateRandomVector(b, d*d, rnd_mat);
  BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
  AccumulateVector resvec;
  auto start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    resvec = bitSerialMatrixVector(bsm, bsv, d);
  auto end = chrono::high_resolution_clock::now();
  float uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  float perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector: " << uscount << " microseconds" << endl;
  cout << "Performance: " << perf << " binary ops per second" << endl;
  // now with thresholding
  ThresholdMatrix T;

  for(size_t l = 0; l < thres_levels; l++) {
    AccumulateVector currentT;
    for(size_t t = 0; t < d; t++) {
      currentT.push_back(rand() % d);
    }
    T.push_back(currentT);
  }
  ResultVector thres_resvec;
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    thres_resvec = bitSerialMatrixVectorThreshold(bsm, bsv, T, d);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector-threshold: " << uscount << " microseconds" << endl;
  cout << "Performance: " << perf << " binary ops per second" << endl;

  delete [] rnd_mat;
  delete [] rnd_vec;
  return 0;
}
