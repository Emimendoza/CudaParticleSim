#ifndef TEST_CUDAGRAVSIM_H
#define TEST_CUDAGRAVSIM_H
#include "VulkanApp.h"
#include <vector>
#include <cstdio>

class CudaGravSim
{
public:
    CudaGravSim(glm::vec3 &cameraPos);

    struct vec3{
        double x;
        double y;
        double z;
    };
    struct Particle
    {
        double mass;
        vec3 pos;
        vec3 vel;
        double size;
    };

    double timeStep = 1000;
    glm::vec3& cameraPos;
    double gravitationalConstant =  1.559*pow(10,(-13)); // ly cubed/(solar mass (year) squared)

    void initArrays(std::vector<VulkanApp::Vertex> vertecies);
    void step();
    void cleanup();
    void (copyData(std::vector<VulkanApp::Vertex>* vertecies));

private:

    Particle* particleArrayDevice;
    Particle* particleArrayHost;
    double* forces;
    uint32_t size;

    void sync();

};

#endif
