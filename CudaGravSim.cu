#include <cmath>

#include "CudaGravSim.h"

__global__ void getAllForces(CudaGravSim::Particle* particles,double** forces, double g, uint32_t size)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>size)
    {
        return;
    }
    CudaGravSim::Particle particle = particles[particleId];
    forces[particleId][0] = 0;
    forces[particleId][1] = 0;
    forces[particleId][2] = 0;
    for(uint32_t index = 0; index<size; index++)
    {
        if (particleId != index)
        {
            CudaGravSim::Particle otherParticle = particles[index];

            double deltaX = particle.pos.x - otherParticle.pos.x;
            double deltaY = particle.pos.y - otherParticle.pos.y;
            double deltaZ = particle.pos.z - otherParticle.pos.z;

            double totalForce = g * (particle.mass * otherParticle.mass) / (pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2));
            // Calculate z comp

            double angle1 = acos(deltaX / (sqrt(pow(deltaX, 2) + pow(deltaZ, 2))));
            double remainder = sin(angle1) * totalForce;
            forces[particleId][2] += totalForce - remainder;
            // Calculate x comp
            double angle2 = acos(deltaX / sqrt(pow(deltaX, 2) + pow(deltaY, 2)));
            totalForce = remainder * sin(angle2);
            forces[particleId][0] += remainder - totalForce;
            // Calculate y comp
            forces[particleId][1] += totalForce;
        }
    }
}

__global__ void getFinalPosition(double** forces, CudaGravSim::Particle* particles, uint32_t size, ulong timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>size)
    {
        return;
    }
    CudaGravSim::Particle particle = particles[particleId];
    particle.pos.x += particle.vel.x*timeStep+(forces[particleId][0]/particle.mass*timeStep*timeStep)/2.0;
    particle.pos.y += particle.vel.y*timeStep+(forces[particleId][1]/particle.mass*timeStep*timeStep)/2.0;
    particle.pos.z += particle.vel.z*timeStep+(forces[particleId][2]/particle.mass*timeStep*timeStep)/2.0;
}

__global__ void getFinalVelocity(double** forces, CudaGravSim::Particle* particles, uint32_t size, ulong timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>size)
    {
        return;
    }
    CudaGravSim::Particle particle = particles[particleId];
    particle.vel.x += forces[particleId][0]/particle.mass*timeStep;
    particle.vel.y += forces[particleId][1]/particle.mass*timeStep;
    particle.vel.z += forces[particleId][2]/particle.mass*timeStep;
}

void CudaGravSim::initArrays(std::vector<VulkanApp::Vertex> vertecies)
{
    size = vertecies.size();
    particleArrayHost = (Particle*) (malloc(sizeof(Particle) * size));
    cudaMalloc(&particleArrayDevice, sizeof(Particle)*size);

    for(uint32_t i = 0; i < vertecies.size(); i++)
    {
        particleArrayHost[i].pos = vertecies[i].pos;
    }
    cudaMemcpy(particleArrayDevice,particleArrayHost,sizeof(Particle)*size,cudaMemcpyHostToDevice);

    cudaMalloc(&forces,sizeof(double)*3*size);
}

void CudaGravSim::cleanup()
{
    free(particleArrayHost);
    cudaFree(forces);
    cudaFree(particleArrayDevice);
}

void CudaGravSim::step()
{
    getAllForces<<<std::ceil(size/1024.0),1024>>>(particleArrayDevice,forces,gravitationalConstant,size);
    getFinalPosition<<<std::ceil(size/1024.0),1024>>>(forces, particleArrayDevice,size,timeStep);
    getFinalVelocity<<<std::ceil(size/1024.0),1024>>>(forces,particleArrayDevice,size,timeStep);
    sync();
}

void CudaGravSim::copyData(std::vector<VulkanApp::Vertex> *vertecies)
{
    std::vector<VulkanApp::Vertex> vert = *vertecies;
    for(uint32_t i = 0; i<size; i++)
    {
        vert[i].pos = particleArrayHost[i].pos;
    }
}

void CudaGravSim::sync()
{
    cudaMemcpy(particleArrayHost,particleArrayDevice,sizeof(Particle)*size,cudaMemcpyDeviceToHost);
}

