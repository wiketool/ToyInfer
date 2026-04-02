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

# for clangd
set(CMAKE_CUDA_FLAGS -g)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

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
        list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/kernel.cu")
        list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/engine.cpp")

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

        if(TARGET CUDA::nvtx3)
            target_link_libraries(ToyInfer PRIVATE CUDA::nvtx3)
        elseif(TARGET CUDA::nvToolsExt)
            target_link_libraries(ToyInfer PRIVATE CUDA::nvToolsExt)
        endif()

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

    add_kernel_test(test_add_bf16_kernel tests/test_add_bf16_kernel.cu)
    add_kernel_test(test_residual_add_bf16 tests/test_residual_add_bf16.cu)
    add_kernel_test(test_precompute_freq_kernel tests/test_precompute_freq_kernel.cu)
    add_kernel_test(test_precompute_freq_f32 tests/test_precompute_freq_f32.cu)
    add_kernel_test(test_reduce_sum_bf16x2_kernel tests/test_reduce_sum_bf16x2_kernel.cu)
    add_kernel_test(test_rmsnorm_bf16x2_kernel tests/test_rmsnorm_bf16x2_kernel.cu)
    add_kernel_test(test_rmsnorm_bf16 tests/test_rmsnorm_bf16.cu)
    add_kernel_test(test_multi_rmsnorm_bf16x2_kernel tests/test_multi_rmsnorm_bf16x2_kernel.cu)
    add_kernel_test(test_multi_rmsnorm_bf16 tests/test_multi_rmsnorm_bf16.cu)
    add_kernel_test(test_gemv_bf16_kernel tests/test_gemv_bf16_kernel.cu)
    add_kernel_test(test_gemv_proj_bf16 tests/test_gemv_proj_bf16.cu)
    add_kernel_test(test_gemv_bf162float_kernel tests/test_gemv_bf162float_kernel.cu)
    add_kernel_test(test_gemv_proj_bf162float tests/test_gemv_proj_bf162float.cu)
    add_kernel_test(test_rope_bf16x2_kernel tests/test_rope_bf16x2_kernel.cu)
    add_kernel_test(test_rope_bf16 tests/test_rope_bf16.cu)
    add_kernel_test(test_cuda_graph_pos tests/test_cuda_graph_pos.cu)
    add_kernel_test(test_gqa_qk_gemv_bf16_kernel tests/test_gqa_qk_gemv_bf16_kernel.cu)
    add_kernel_test(test_softmax_f32_kernel tests/test_softmax_f32_kernel.cu)
    add_kernel_test(test_apply_score2v_f32_kernel tests/test_apply_score2v_f32_kernel.cu)
    add_kernel_test(test_convert_float22bfloat162 tests/test_convert_float22bfloat162.cu)
    add_kernel_test(test_attention_bf16 tests/test_attention_bf16.cu)
    add_kernel_test(test_flash_attention_v1_bf16 tests/test_flash_attention_v1_bf16.cu)
    add_kernel_test(test_swiglu_bf16x2_kernel tests/test_swiglu_bf16x2_kernel.cu)
    add_kernel_test(test_swiglu_bf16x2 tests/test_swiglu_bf16x2.cu)
    add_kernel_test(test_batch_gemv_proj_bf16 tests/test_batch_gemv_proj_bf16.cu)
    add_kernel_test(test_batch_rmsnorm_bf16 tests/test_batch_rmsnorm_bf16.cu)
    add_kernel_test(test_batch_multi_rmsnorm_bf16 tests/test_batch_multi_rmsnorm_bf16.cu)
    add_kernel_test(test_batch_residual_add_bf16 tests/test_batch_residual_add_bf16.cu)
    add_kernel_test(test_batch_swiglu_bf16x2 tests/test_batch_swiglu_bf16x2.cu)

    if(TOYINFER_APP_DEPS_READY)
        add_executable(test_tokenizer
            tests/test_tokenizer.cpp
            src/config.cpp
            src/tokenizer.cpp
        )
        target_include_directories(test_tokenizer PRIVATE include)
        target_compile_definitions(test_tokenizer PRIVATE
            TOYINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )
        target_link_libraries(test_tokenizer PRIVATE
            CLI11::CLI11
            spdlog::spdlog
            jsoncpp_static
        )
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_tokenizer PRIVATE -fsanitize=address)
            target_link_options(test_tokenizer PRIVATE -fsanitize=address)
        endif()
        add_test(NAME test_tokenizer COMMAND test_tokenizer)
    endif()
endif()
