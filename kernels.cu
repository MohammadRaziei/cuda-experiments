#include <chrono>
#include <iostream>

#include <ppl.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "io.h"
#include "thrust/sort.h"

#ifdef __INTELLISENSE__
#define __syncthreads()
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#define MaxMedianWindowSize (300)

/// function to sort the array in ascending order
template <typename T>
__device__ __forceinline__ void arraySort(T* array, const uint32_t n) {
  /// declare some local variables
  T temp = 0;
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n - 1; j++) {
      if (array[j] > array[j + 1]) {
        temp = array[j];
        array[j] = array[j + 1];
        array[j + 1] = temp;
      }
    }
  }
}

// KERNEL: Median Filter
__device__ void sortedInOut(float sortedData[], const uint32_t len,
                            const float& outValue, const float& inValue) {
  bool notFound = true;
  float value = sortedData[0], saveValue;
  sortedData[len] = inValue;
  for (uint32_t j = 0, i = 0; i < len; ++i) {
    if (value == outValue) {
      value = sortedData[++j];
    }
    if (notFound && value >= inValue) {
      sortedData[i] = inValue;
      notFound = false;
    } else {
      saveValue = value;
      value = sortedData[++j];
      sortedData[i] = saveValue;
    }
  }
}
__device__ void medianFilterCore(float output[], const float input[],
                                 const uint32_t len,
                                 const uint32_t halfWindow) {
  const uint32_t windowSize = 2 * halfWindow + 1;
  float temp[MaxMedianWindowSize];
  {
    float* temp_p = temp;
    const float* input_p = input;
    for (uint32_t i = 0; i < windowSize; ++i) *(temp_p++) = *(input_p++);
  }
  arraySort(temp, windowSize);
  output[0] = temp[halfWindow];
  for (uint32_t i = 0; i < len - 1; ++i) {
    sortedInOut(temp, windowSize, input[i], input[i + windowSize]);
    output[i + 1] = temp[halfWindow];
  }
}
__global__ void medianFilterKernel(float output[], const float input[],
                                   const uint32_t len,
                                   const uint32_t halfWindow) {
  const uint32_t numFrames = blockDim.x * gridDim.x;
  const uint32_t lenFrames = (uint32_t)ceilf((float)len / numFrames);
  const uint32_t i = (threadIdx.x + blockDim.x * blockIdx.x) * lenFrames;
  if (i < len) {
    if (i + lenFrames <= len)
      medianFilterCore(output + i, input + i, lenFrames, halfWindow);
    else
      medianFilterCore(output + i, input + i, len - i, halfWindow);
  }
}

// KERNEL: Abs
__global__ void absKernel(float* out, float* real, float* imag, uint32_t size) {
  for (uint32_t i{threadIdx.x + blockIdx.x * blockDim.x}; i < size;
       i += blockDim.x * gridDim.x)
    out[i] = hypotf(real[i], imag[i]);
}

// KERNEL: Subtract
__global__ void mysubtractKernel(float out[], const float in1[],
                                 const float in2[], const uint32_t len) {
  for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    out[i] = in1[i] - in2[i];
  }
}

// KERNEL: sqrHypot
__global__ void sqrHypotKernel(float out[], const float sigReal[],
                               const float sigImag[], const uint32_t len) {
  for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    out[i] = sigReal[i] * sigReal[i] + sigImag[i] * sigImag[i];
  }
}
__global__ void sqrHypotKernel(float out[], const cuComplex sigComp[],
                               const uint32_t len) {
  for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    out[i] = sigComp[i].x * sigComp[i].x + sigComp[i].y * sigComp[i].y;
  }
}
__global__ void sqrHypotKernel(cuComplex out[], const cuComplex sigComp[],
                               const uint32_t len) {
  for (uint32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    out[i].x = sigComp[i].x * sigComp[i].x + sigComp[i].y * sigComp[i].y;
  }
}

// KERNEL: sum
__global__ void sumSerialkernel(float* out, const float input[],
                                const uint32_t len, float scale = 1.f) {
  if (0 == (threadIdx.x + blockDim.x * blockIdx.x)) {
    float sumVal = 0;
    for (uint32_t i = 0; i < len; ++i) {
      sumVal += input[i];
    }
    *out = sumVal / scale;
  }
}