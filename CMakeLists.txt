cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
project(Test LANGUAGES CUDA CXX)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-ccbin gcc-11")

find_package(Vulkan COMPONENTS glslc)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

function(compile_shader target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "ENV;FORMAT" "SOURCES")
    foreach(source ${arg_SOURCES})
        add_custom_command(
                OUTPUT ${source}.${arg_FORMAT}
                DEPENDS ${source}
                DEPFILE ${source}.d
                COMMAND
                ${glslc_executable}
                $<$<BOOL:${arg_ENV}>:--target-env=${arg_ENV}>
                $<$<BOOL:${arg_FORMAT}>:-mfmt=${arg_FORMAT}>
                -MD -MF ${source}.d
                -o ${source}.${arg_FORMAT}
                ${CMAKE_CURRENT_SOURCE_DIR}/${source}
        )
        target_sources(${target} PRIVATE ${source}.${arg_FORMAT})
    endforeach()
endfunction()


add_executable(Test ./src/main.cu ./src/VulkanApp.h ./src/VulkanApp.cpp)

compile_shader(Test
        ENV vulkan1.2
        FORMAT bin
        SOURCES
            ./src/shaders/shader.frag
            ./src/shaders/shader.vert
        )

target_link_libraries(Test -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi)

set_target_properties(Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
