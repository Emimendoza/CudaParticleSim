#include <cmath>

#include "CudaGravSim.h"

__global__ void initForces(CudaGravSim::vec3* force, CudaGravSim::vec3* force2, uint32_t size)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    force[particleId] = {0,0,0};
    force2[particleId] = {0,0,0};
}

__global__ void getAllForces(CudaGravSim::Particle* particles, CudaGravSim::vec3* forces, float g, uint32_t size, CudaGravSim::vec3 cameraPos)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    forces[particleId] = {0,0,0};
    for(uint32_t index = 0; index<size; index++)
    {
        if (particleId != index)
        {
            CudaGravSim::vec3 delta = particles[particleId].pos - particles[index].pos;

            float absoluteDistance = sqrt(delta.square().sum());
            float totalForce = g * (particles[particleId].mass * particles[index].mass) /pow(absoluteDistance,3);

            forces[index] += delta*totalForce;
        }
    }
    particles[particleId].size=4/(particles[particleId].pos-cameraPos).square().sum();
}

__global__ void getFinalPosition(CudaGravSim::vec3* forces, CudaGravSim::Particle* particles, uint32_t size, float timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    float time = pow(timeStep,2);
    particles[particleId].pos += particles[particleId].vel*timeStep+(forces[particleId]*time/(particles[particleId].mass*2));
}

__global__ void getFinalVelocity(CudaGravSim::vec3 *forces,CudaGravSim::vec3* forces2, CudaGravSim::Particle* particles, uint32_t size, float timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    particles[particleId].vel += (forces[particleId]+forces2[particleId])/(particles[particleId].mass*2)*timeStep;
}

void CudaGravSim::initArrays(std::vector<VulkanApp::Vertex> vertecies)
{
    size = vertecies.size();
    cudaMalloc(&particleArrayDevice,sizeof(Particle)*size);
    cudaMalloc(&forces,sizeof(vec3)*size);
    cudaMalloc(&forces2, sizeof(vec3)*size);
    particleArrayHost = static_cast<Particle *>(malloc(sizeof(Particle) * size));
    for(uint32_t i = 0; i<size; i++)
    {
        particleArrayHost[i].pos = vertecies[i].pos;
        particleArrayHost[i].mass = 1;
        particleArrayHost[i].vel = {0,0,0};
    }
    particleArrayHost[0].pos = {0.25,0,0.25};
    particleArrayHost[0].mass = 10;
    particleArrayHost[0].vel= {0,0,0};
    cudaMemcpy(particleArrayDevice,particleArrayHost,size*sizeof(Particle),cudaMemcpyHostToDevice);
    initForces<<<ceil(size/1024.0),1024>>>(forces, forces2, size);
}

void CudaGravSim::cleanup()
{
    free(particleArrayHost);
    cudaFree(particleArrayDevice);
    cudaFree(forces);
    cudaFree(forces2);
}

void CudaGravSim::step()
{
    if(force1)
    {
        getFinalPosition<<<std::ceil(size/1024.0),1024>>>(forces2, particleArrayDevice,size,timeStep);
        getAllForces<<<std::ceil(size/1024.0),1024>>>(particleArrayDevice,forces,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<std::ceil(size/1024.0),1024>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = false;
    }
    else
    {
        getFinalPosition<<<std::ceil(size/1024.0),1024>>>(forces, particleArrayDevice,size,timeStep);
        getAllForces<<<std::ceil(size/1024.0),1024>>>(particleArrayDevice,forces2,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<std::ceil(size/1024.0),1024>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = true;
    }
    sync();
}

void CudaGravSim::copyData(std::vector<VulkanApp::Vertex> *vertecies)
{
    std::vector<VulkanApp::Vertex>& vert = *vertecies;
    for(uint32_t i = 0; i<size; i++)
    {
        vert[i].pos = {particleArrayHost[i].pos.x,particleArrayHost[i].pos.y,particleArrayHost[i].pos.z,particleArrayHost[i].size};
    }
}

void CudaGravSim::sync()
{
    cudaMemcpy(particleArrayHost,particleArrayDevice,size*sizeof(Particle),cudaMemcpyDeviceToHost);
}

CudaGravSim::CudaGravSim(glm::vec3 &cameraPos2)
{
    cameraPos = cameraPos2;
}
