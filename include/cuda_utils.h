#pragma once

#include <cuda_runtime.h>

#include <iostream>

// 核心检查宏：使用 do-while(0) 包裹是为了在 if-else 语句中安全使用
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error:\n"                                       \
                      << "    File:       " << __FILE__ << "\n"                \
                      << "    Line:       " << __LINE__ << "\n"                \
                      << "    Error code: " << err << "\n"                     \
                      << "    Error text: " << cudaGetErrorString(err) << "\n" \
                      << "    Function:   " << #call << std::endl;             \
            /* 根据需要选择是退出程序还是抛出异常 */                           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 专门用于检查核函数 (Kernel) 启动后的错误
#define CHECK_LAST_CUDA_ERROR()                                        \
    do {                                                               \
        cudaError_t err = cudaGetLastError();                          \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA Kernel Error:\n"                        \
                      << "    File:       " << __FILE__ << "\n"        \
                      << "    Line:       " << __LINE__ << "\n"        \
                      << "    Error code: " << err << "\n"             \
                      << "    Error text: " << cudaGetErrorString(err) \
                      << std::endl;                                    \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)