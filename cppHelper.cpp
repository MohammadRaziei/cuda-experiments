#include <stdio.h>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define cat(x, y) x##y
#define seeType(TYPE, arr, len) \
  std::vector<TYPE> cat(seeVec_, arr)(arr, arr + len)
#define seeComplex16(arr, len) seeType(complex16, arr, len)
#define seeFloat(arr, len) seeType(float, arr, len)
#define sendToMatlab(arr, len) writeToFile(std::string(#arr) + ".bin", arr, len)

// Tools: printDev
#define showDev(arr, len) \
  printf("%s: ", #arr);   \
  printDev(arr, len);

void printDev(const float* dev_arr, const uint32_t len = 1) {
  for (uint32_t i = 0; i < (uint32_t)min(10, len); ++i)
    printf("%f, ", dev_arr[i]);
  printf("\n");
}
template <typename T>
void writeToFile(const std::string& filename, T* data, uint32_t len) {
  std::ofstream outFile(filename, std::ios::out | std::ofstream::binary);
  if (!outFile.is_open()) {
    fprintf(stderr, "Cannot open: \"%s\"\n", filename.c_str());
    return;
  }
  outFile.write(reinterpret_cast<const char*>(data), sizeof(T) * len);
  outFile.close();
}
template <typename T>
inline void writeToFile(const std::string& filename, std::vector<T>* data) {
  writeToFile(filename, data.data, static_cast<T>(data.size()));
}