#include "kernel_warpper.h"

namespace toyinfer {
__global__ void add_kernel(float* a, float* b, float* c, int n) {}

void launch_add_kernel(float* a, float* b, float* c, int n) {
    add_kernel<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
};
}  // namespace toyinfer