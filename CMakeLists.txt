cmake_minimum_required(VERSION 3.20)

project(ToyInfer LANGUAGES CXX C CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures:
# 75=Turing (20xx), 
# 86=Ampere (30xx), 
# 89=Ada Lovelace (40xx), 
# 90=Hopper
# set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)
set(CMAKE_CUDA_ARCHITECTURES 89)

message(STATUS "========================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
option(BUILD_TOYINFER_APP "Build ToyInfer executable" ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Set up ccache ...")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

include_directories(include)

if(BUILD_TOYINFER_APP)
    set(TOYINFER_APP_DEPS_READY TRUE)
    foreach(dep IN ITEMS spdlog cli11 jsoncpp)
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/${dep}/CMakeLists.txt")
            message(WARNING
                "Missing third_party/${dep}. Run `git submodule update --init --recursive` "
                "to build ToyInfer executable.")
            set(TOYINFER_APP_DEPS_READY FALSE)
        endif()
    endforeach()

    if(TOYINFER_APP_DEPS_READY)
        add_subdirectory(third_party/linenoise)
        add_subdirectory(third_party/spdlog)
        add_subdirectory(third_party/cli11)
        add_subdirectory(third_party/jsoncpp)

        file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
            "src/*.cpp"
            "src/*.c"
            "src/*.cu"
        )

        add_executable(ToyInfer
            main.cpp
            ${SRC_FILES}
        )

        target_link_libraries(
            ToyInfer
            PRIVATE
            CUDA::cublas
            CLI11::CLI11
            spdlog::spdlog
            jsoncpp_static
            linenoise
        )

        target_precompile_headers(ToyInfer PRIVATE
            <spdlog/spdlog.h>
            <CLI/CLI.hpp>
            <json/json.h>
            <linenoise.h>
        )

        target_compile_options(ToyInfer PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )

        # disable all warnings for both C++ and CUDA
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(ToyInfer PRIVATE -w)
        endif()

        if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
            target_compile_options(ToyInfer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
        endif()
    endif()
endif()

if(BUILD_TESTING)
    function(add_kernel_test test_name test_src)
        add_executable(${test_name} ${test_src})
        target_include_directories(${test_name} PRIVATE include tests)
        target_link_libraries(${test_name} PRIVATE CUDA::cudart)
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    add_kernel_test(test_precompute_freq tests/test_precompute_freq.cu)
    add_kernel_test(test_rmsnorm tests/test_rmsnorm.cu)
    add_kernel_test(test_multi_rmsnorm tests/test_multi_rmsnorm.cu)
    add_kernel_test(test_attn_single_proj tests/test_attn_single_proj.cu)
    add_kernel_test(test_rope tests/test_rope.cu)
endif()
