LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := benchmark
LOCAL_SRC_FILES := ../benchmark/benchmark.cpp
#LOCAL_CFLAGS += -save-temps
include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test/test.cpp
LOCAL_CFLAGS += -UNDEBUG 
include $(BUILD_EXECUTABLE)
