#include <time.h>
#include <cstdlib>
#include <deque>
#include "gemm-bitserial.h"

using namespace std;

#define VERBOSE_TEST(x) ;
//#define VERBOSE_TEST(x) x

bool test_conversions() {
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
      VERBOSE_TEST(cout << "Bits = " << b << " dim = " << d << " result = " << res << endl);
    }
  }
  cout << "Conversion tests: " << ok << " OK, " << nok << " NOK" << endl;
  return ok == numConfigs;
}

bool test_matrix_vector() {
  vector<size_t> param_bits {1, 2, 3, 4, 5, 6, 7, 8};
  vector<size_t> param_dims {16, 32, 100, 256};
  deque<bool> param_allow_neg {false, true};
  unsigned int numConfigs = 0, ok = 0, nok = 0;
  for(auto & b: param_bits) {
    for(auto & d: param_dims) {
      for(auto & matneg: param_allow_neg) {
        for(auto & vecneg: param_allow_neg) {
          uint8_t * rnd_mat = new uint8_t[d*d];
          uint8_t * rnd_vec = new uint8_t[d];
          ResultVector res_golden;
          generateRandomVector(b, d, rnd_vec);
          generateRandomVector(b, d*d, rnd_mat);
          BitSerialVector bsv = toBitSerialVector(rnd_vec, d, b);
          BitSerialMatrix bsm = toBitSerialMatrix(rnd_mat, d, d, b);
          ResultVector resvec = bitSerialMatrixVector(bsm, bsv, d, matneg, vecneg);
          // manually compute golden result
          for(unsigned int i = 0; i < d; i++) {
            ResultElem acc = 0;
            for(unsigned int j = 0; j < d; j++) {
              int32_t matelem_adj = rnd_mat[i * d + j];
              int32_t vecelem_adj = rnd_vec[j];
              if(matneg && ((matelem_adj & (1 << (b-1))) != 0) ) {
                matelem_adj -= 1 << b;
              }
              if(vecneg && ((vecelem_adj & (1 << (b-1))) != 0) ) {
                vecelem_adj -= 1 << b;
              }
              acc += matelem_adj * vecelem_adj;
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
          VERBOSE_TEST(cout << "Bits = " << b << " dim = " << d << " matneg = " << matneg << " vecneg = " << vecneg << " result = " << res << endl);
        }
      }
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
