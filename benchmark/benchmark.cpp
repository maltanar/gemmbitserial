#include <cassert>
#include <iostream>
#include <chrono>
#include "gemm-bitserial.h"

using namespace std;

int main(int argc, char const *argv[]) {
  size_t b = 2;
  size_t d = 1024;
  size_t thres_levels = 3;
  size_t reps = 10000;
  size_t reps_mm = 10;
  uint8_t * rnd_mat = new uint8_t[d*d];
  uint8_t * rnd_vec = new uint8_t[d];
  assert(rnd_mat != 0 && rnd_vec != 0);
  generateRandomVector(b, d, rnd_vec);
  generateRandomVector(b, d*d, rnd_mat);
  BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
  BitSerialVector bsv2 = toBitSerialVector(rnd_vec, d, b);
  BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
  // and_cardinality
  auto start = chrono::high_resolution_clock::now();
  uint64_t res_card;
  for(unsigned int i = 0; i < reps; i++)
    res_card = bsv[0].and_cardinality(bsv2[0]);
  auto end = chrono::high_resolution_clock::now();
  float uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  float perf = 1000000 * (d*2 / uscount);
  cout << "Time for and_cardinality: " << uscount << " microseconds" << endl;
  cout << "Performance for and_cardinality: " << perf/1000000000.0 << " GOPS per second" << endl;
  // matrix-vector
  AccumulateVector resvec;
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    resvec = bitSerialMatrixVector(bsm, bsv);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-vector: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-vector: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;
  // matrix-matrix
  start = chrono::high_resolution_clock::now();
  AccumulateMatrix resmat;
  for(unsigned int i = 0; i < reps_mm; i++)
    resmat = bitSerialMatrixMatrix(bsm, bsm);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps_mm;
  perf = 1000000 * (d*d*d*b*b*2 / uscount);
  cout << "Time for single matrix-matrix: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-matrix: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-matrix: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;
  // matrix-vector-threshold
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
    thres_resvec = bitSerialMatrixVectorThreshold(bsm, bsv, T);
  end = chrono::high_resolution_clock::now();
  uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  perf = 1000000 * (d*d*b*b*2 / uscount);
  cout << "Time for single matrix-vector-threshold: " << uscount << " microseconds" << endl;
  cout << "Performance for matrix-vector-threshold, not counting threshold ops: " << perf/1000000000.0 << " binary GOPS per second" << endl;
  cout << "Performance for matrix-vector-threshold, not counting threshold ops: " << perf/(b*b*1000000000.0) << " lowprec (" << b << " bit) GOPS per second" << endl;

  delete [] rnd_mat;
  delete [] rnd_vec;
  return 0;
}
