cmake_minimum_required(VERSION 3.21)
project(Test CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
project(Test LANGUAGES CUDA CXX)


add_executable(Test main.cu VulkanApp.h VulkanApp.cpp)

target_link_libraries(Test -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi)

set_target_properties(Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
