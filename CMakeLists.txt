cmake_minimum_required(VERSION 3.20)

project(ToyInfer LANGUAGES CXX C CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures:
# 75=Turing (20xx), 
# 86=Ampere (30xx), 
# 89=Ada Lovelace (40xx), 
# 90=Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

message(STATUS "========================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(third_party/linenoise)
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/cli11)
add_subdirectory(third_party/jsoncpp)
include_directories(src)

add_executable(
    ToyInfer
    main.cu
)

target_link_libraries(
    ToyInfer 
    PRIVATE
    CUDA::cublas
    CLI11::CLI11
    spdlog::spdlog
    jsoncpp:jsoncpp
    linenoise
)

target_compile_options(ToyInfer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(ToyInfer PROPERTIES COMPILE_FLAGS "-O3")

# disable all warnings for both C++ and CUDA
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ToyInfer PRIVATE -w)
endif()

if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(ToyInfer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
endif()
