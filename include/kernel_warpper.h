#pragma once
#include <cuda_runtime.h>

namespace toyinfer {
void launch_add_kernel(float* a, float* b, float* c, int n);
}