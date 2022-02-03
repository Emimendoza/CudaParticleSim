#include <cmath>

#include "CudaGravSim.h"

__global__ void getAllForces(CudaGravSim::Particle* particles,double* forces, double g, uint32_t size)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    forces[particleId*3+0] = 0;
    forces[particleId*3+1] = 0;
    forces[particleId*3+2] = 0;
    for(uint32_t index = 0; index<size; index++)
    {
        if (particleId != index)
        {

            double deltaX = particles[particleId].pos.x - particles[index].pos.x;
            double deltaY = particles[particleId].pos.y - particles[index].pos.y;
            double deltaZ = particles[particleId].pos.z - particles[index].pos.z;

            double absoluteDistance = sqrt(pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2));
            double totalForce = g * (particles[particleId].mass * particles[index].mass) /pow(absoluteDistance,3);

            forces[index*3+0] += totalForce*deltaX;
            forces[index*3+1] += totalForce*deltaY;
            forces[index*3+2] += totalForce*deltaZ;
        }
    }
}

__global__ void getFinalPosition(double* forces, CudaGravSim::Particle* particles, uint32_t size, double timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    double time = pow(timeStep,2);
    particles[particleId].pos.x += particles[particleId].vel.x*timeStep+(forces[particleId*3+0]/particles[particleId].mass*time)/2.0;
    particles[particleId].pos.y += particles[particleId].vel.y*timeStep+(forces[particleId*3+1]/particles[particleId].mass*time)/2.0;
    particles[particleId].pos.z += particles[particleId].vel.z*timeStep+(forces[particleId*3+2]/particles[particleId].mass*time)/2.0;
}

__global__ void getFinalVelocity(double* forces, CudaGravSim::Particle* particles, uint32_t size, double timeStep)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=size)
    {
        return;
    }
    particles[particleId].vel.x += forces[particleId*3+0]/particles[particleId].mass*timeStep;
    particles[particleId].vel.y += forces[particleId*3+1]/particles[particleId].mass*timeStep;
    particles[particleId].vel.z += forces[particleId*3+2]/particles[particleId].mass*timeStep;
}

void CudaGravSim::initArrays(std::vector<VulkanApp::Vertex> vertecies)
{
    size = vertecies.size();
    cudaMalloc(&particleArrayDevice,sizeof(Particle)*size);
    cudaMalloc(&forces,sizeof(double)*size*3);
    particleArrayHost = static_cast<Particle *>(malloc(sizeof(Particle) * size));
    for(uint32_t i = 0; i<size; i++)
    {
        particleArrayHost[i].pos = vertecies[i].pos;
        particleArrayHost[i].mass = 10;
        particleArrayHost[i].vel = {0,0,0};
    }
    cudaMemcpy(particleArrayDevice,particleArrayHost,size*sizeof(Particle),cudaMemcpyHostToDevice);
}

void CudaGravSim::cleanup()
{
    free(particleArrayHost);
    cudaFree(particleArrayDevice);
    cudaFree(forces);
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
    std::vector<VulkanApp::Vertex>& vert = *vertecies;
    for(uint32_t i = 0; i<size; i++)
    {
        vert[i].pos = particleArrayHost[i].pos;
    }
}

void CudaGravSim::sync()
{
    double* temp;
    temp = (double*) malloc(sizeof(double )*3*size);
    cudaMemcpy(particleArrayHost,particleArrayDevice,size*sizeof(Particle),cudaMemcpyDeviceToHost);
    cudaMemcpy(temp,forces,size*sizeof(double)*3,cudaMemcpyDeviceToHost);
    free(temp);
}