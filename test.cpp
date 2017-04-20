#include <time.h>
#include <cstdlib>
#include "gemm-bitserial.h"

using namespace std;

void generateRandomVector(size_t bits, size_t dim, uint8_t * ret) {
  uint8_t minVal = 0;
  uint8_t maxVal = (1 << bits) - 1;
  for(size_t i = 0; i < dim; i++) {
    ret[i] = rand() % maxVal;
  }
}

bool test_conversions() {
  cout << "Now running conversion tests..." << endl;
  vector<size_t> param_bits {1, 2, 3, 4, 5, 6, 7, 8};
  vector<size_t> param_dims {16, 32, 100, 1024, 4096};
  unsigned int numConfigs = 0, ok = 0, nok = 0;

  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      uint8_t * res_chk = new uint8_t[d];
      uint8_t * rnd_vec = new uint8_t[d];
      generateRandomVector(b, d, rnd_vec);
      BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
      fromBitSerialVector(bsv, d, res_chk);
      int res = memcmp(rnd_vec, res_chk, d);
      delete [] rnd_vec;
      delete [] res_chk;
      if(res == 0) {
        ok++;
      } else {
        nok++;
      }
      numConfigs++;
      cout << "Bits = " << b << " dim = " << d << " result = " << res << endl;
    }
  }
  cout << "Conversion tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_matrix_vector() {
  cout << "Now running matrix vector multiplication tests..." << endl;
  vector<size_t> param_bits {1, 2, 3, 4, 5, 6, 7, 8};
  vector<size_t> param_dims {16, 32, 100, 1024, 4096};
  unsigned int numConfigs = 0, ok = 0, nok = 0;
  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      uint8_t * rnd_mat = new uint8_t[d*d];
      uint8_t * rnd_vec = new uint8_t[d];
      ResultVector res_golden;
      generateRandomVector(b, d, rnd_vec);
      generateRandomVector(b, d*d, rnd_mat);
      BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
      BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
      ResultVector resvec = bitSerialMatrixVector(bsm, bsv, d);
      // manually compute golden result
      for(unsigned int i = 0; i < d; i++) {
        ResultElem acc = 0;
        for(unsigned int j = 0; j < d; j++) {
          acc += rnd_mat[i * d + j] * rnd_vec[j];
        }
        res_golden.push_back(acc);
      }
      int res = (res_golden == resvec) ? 0 : 1;
      delete [] rnd_vec;
      delete [] rnd_mat;
      if(res == 0) {
        ok++;
      } else {
        nok++;
      }
      numConfigs++;
      cout << "Bits = " << b << " dim = " << d << " result = " << res << endl;
    }
  }
  cout << "Matrix vector multiplication tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}


int main(int argc, char const *argv[]) {
  srand(time(NULL));
  bool all_ok = true;
  all_ok &= test_conversions();
  all_ok &= test_matrix_vector();

  if(all_ok) {
    cout << "All tests completed successfully" << endl;
  } else {
    cout << "Some tests failed" << endl;
  }
  return 0;
}
