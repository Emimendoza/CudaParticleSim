cmake_minimum_required(VERSION 3.21)
project(Test CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Test main.cu VulkanApp.cpp VulkanApp.h)


target_link_libraries(Test -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi)

set_target_properties(Test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
