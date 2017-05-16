LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := benchmark
LOCAL_SRC_FILES := ../benchmark/benchmark.cpp ../bitvector.cpp ../cnpy.cpp ../convert.cpp ../gemm-bitserial.cpp ../serialize.cpp

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test/test.cpp ../bitvector.cpp ../cnpy.cpp ../convert.cpp ../gemm-bitserial.cpp ../serialize.cpp

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := example-sfc
LOCAL_SRC_FILES := ../example-sfc/sfc.cpp ../bitvector.cpp ../cnpy.cpp ../convert.cpp ../gemm-bitserial.cpp ../serialize.cpp

include $(BUILD_EXECUTABLE)
