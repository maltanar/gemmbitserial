
#include "gemm-bitserial.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "indata.h"
using namespace std;

// network params as global variables

ThresholdMatrix CPUThresholdLayer_0_T;
BitSerialMatrix CPUBitSerialMatrixVectorThresholdLayer_2_W;
ThresholdMatrix CPUBitSerialMatrixVectorThresholdLayer_2_T;
BitSerialMatrix CPUBitSerialMatrixVectorThresholdLayer_4_W;
ThresholdMatrix CPUBitSerialMatrixVectorThresholdLayer_4_T;
BitSerialMatrix CPUBitSerialMatrixVectorThresholdLayer_6_W;
ThresholdMatrix CPUBitSerialMatrixVectorThresholdLayer_6_T;
BitSerialMatrix CPUBitSerialMatrixVectorLayer_8_W;
FloatVector CPULinearLayer_10_A, CPULinearLayer_10_B;

// naive LinearLayer implementation, int in, float params and out
FloatVector linearLayer(const AccumulateVector & in, const FloatVector & mul, const FloatVector & add) {
  FloatVector ret;
  size_t scale_param_ind = 0;
  const size_t scale_param_wraparound = mul.size() - 1;
  for(size_t i = 0; i < in.size(); i++) {
    ret.push_back(in[i] * mul[scale_param_ind] + add[scale_param_ind]);
    scale_param_ind = (scale_param_ind == scale_param_wraparound) ? 0 : (scale_param_ind+1);
  }
  return ret;
}

FloatVector pipeline(const AccumulateVector & data_in) {
  // neural network pipeline
  
  // layer: CPUThresholdLayer_0
  ResultVector BufferLayer_1 = threshold(data_in, CPUThresholdLayer_0_T);
  // layer: CPUBitSerialMatrixVectorThresholdLayer_2
  BitSerialVector BufferLayer_1_bs =  toBitSerialVector(BufferLayer_1.data(), BufferLayer_1.size(), 2);
  ResultVector BufferLayer_3 = bitSerialMatrixVectorThreshold(CPUBitSerialMatrixVectorThresholdLayer_2_W, BufferLayer_1_bs, CPUBitSerialMatrixVectorThresholdLayer_2_T, true, false);
  
  // layer: CPUBitSerialMatrixVectorThresholdLayer_4
  BitSerialVector BufferLayer_3_bs =  toBitSerialVector(BufferLayer_3.data(), BufferLayer_3.size(), 2);
  ResultVector BufferLayer_5 = bitSerialMatrixVectorThreshold(CPUBitSerialMatrixVectorThresholdLayer_4_W, BufferLayer_3_bs, CPUBitSerialMatrixVectorThresholdLayer_4_T, true, false);
  
  // layer: CPUBitSerialMatrixVectorThresholdLayer_6
  BitSerialVector BufferLayer_5_bs =  toBitSerialVector(BufferLayer_5.data(), BufferLayer_5.size(), 2);
  ResultVector BufferLayer_7 = bitSerialMatrixVectorThreshold(CPUBitSerialMatrixVectorThresholdLayer_6_W, BufferLayer_5_bs, CPUBitSerialMatrixVectorThresholdLayer_6_T, true, false);
  
  // layer: CPUBitSerialMatrixVectorLayer_8
  BitSerialVector BufferLayer_7_bs =  toBitSerialVector(BufferLayer_7.data(), BufferLayer_7.size(), 2);
  AccumulateVector BufferLayer_9 = bitSerialMatrixVector(CPUBitSerialMatrixVectorLayer_8_W, BufferLayer_7_bs, true, false);
  
  // layer: CPULinearLayer_10
  FloatVector data_out = linearLayer(BufferLayer_9, CPULinearLayer_10_A, CPULinearLayer_10_B);
  
  return data_out;
}

cnpy::NpyArray loadNpzParam(string vname) {
  return cnpy::npz_load("params.npz", vname);
}

int main(int argc, char const *argv[]) {
  // prepare network parameters
  
  CPUThresholdLayer_0_T = toThresholdMatrix(loadNpzParam("CPUThresholdLayer_0_T"));
  CPUBitSerialMatrixVectorThresholdLayer_2_W = toBitSerialMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_2_W"), 2);
  CPUBitSerialMatrixVectorThresholdLayer_2_T = toThresholdMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_2_T"));
  CPUBitSerialMatrixVectorThresholdLayer_4_W = toBitSerialMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_4_W"), 2);
  CPUBitSerialMatrixVectorThresholdLayer_4_T = toThresholdMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_4_T"));
  CPUBitSerialMatrixVectorThresholdLayer_6_W = toBitSerialMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_6_W"), 2);
  CPUBitSerialMatrixVectorThresholdLayer_6_T = toThresholdMatrix(loadNpzParam("CPUBitSerialMatrixVectorThresholdLayer_6_T"));
  CPUBitSerialMatrixVectorLayer_8_W = toBitSerialMatrix(loadNpzParam("CPUBitSerialMatrixVectorLayer_8_W"), 2);
  CPULinearLayer_10_A = toFloatVector(loadNpzParam("CPULinearLayer_10_A"));
  CPULinearLayer_10_B = toFloatVector(loadNpzParam("CPULinearLayer_10_B"));
  int reps = 100;
  vector<float> out;

  auto start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < reps; i++)
    out = pipeline(indata);
  auto end = chrono::high_resolution_clock::now();
  float uscount = chrono::duration_cast<std::chrono::microseconds>(end-start).count() / (float)reps;
  float fps = 1000000 / uscount;
  cout << "Time for SFC inference: " << uscount << " microseconds" << endl;
  cout << "Frames per second: " << fps << endl;

  cout << "Returned result vector: " << endl;
  cout << out << endl;

  return 0;
}
