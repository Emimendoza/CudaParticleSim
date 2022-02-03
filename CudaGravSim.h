#ifndef TEST_CUDAGRAVSIM_H
#define TEST_CUDAGRAVSIM_H
#include "VulkanApp.h"
#include <vector>
#include <cstdio>

class CudaGravSim
{
public:
    struct Particle
    {
        double mass = 1000000000;
        glm::vec3 pos;
        glm::vec3 vel;
    };

    ulong timeStep = 1000;
    double gravitationalConstant = 1; //6.67408 * pow(10,-11);

    void initArrays(std::vector<VulkanApp::Vertex> vertecies);
    void step();
    void cleanup();
    void (copyData(std::vector<VulkanApp::Vertex>* vertecies));

private:

    Particle* particleArrayDevice;
    Particle* particleArrayHost;
    double** forces;
    uint32_t size;

    void sync();

};

#endif
