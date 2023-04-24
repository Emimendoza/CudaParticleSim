// Main.cu
#include <iostream>
#include "VulkanApp.h"
#define THREADS_PER_KERNEL 1024
// Vulkan Interop Code from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html - 3.2.16.1
int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice)
{
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice, &vkPhysicalDeviceProperties2);

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

// Vector Math
struct mephvec3{

    __host__ __device__ mephvec3()
    {
        x=0;
        y=0;
        z=0;
    }
    __host__ __device__ mephvec3(float x1, float y1, float z1)
    {
        x=x1;
        y=y1;
        z=z1;
    }

    float x;
    float y;
    float z;

    __device__ __host__ mephvec3& operator=(glm::vec3& vec1)
    {
        this->x = vec1.x;
        this->y = vec1.y;
        this->z = vec1.z;
        return *this;
    }

    __device__ __host__ mephvec3& operator=(const glm::vec4& vec1)
    {
        this->x = vec1.x;
        this->y = vec1.y;
        this->z = vec1.z;
        return *this;
    }

    __device__ __host__ mephvec3 operator*(const mephvec3& other) const
    {
        return {
            other.x*this->x,
            other.y*this->y,
            other.z*this->z
        };
    }
    __device__ __host__ mephvec3 operator*(const float& other) const
    {
        return {
                other * this->x,
                other * this->y,
                other * this->z
        };
    }

    __device__ __host__ mephvec3 operator/(const float& other) const
    {
        return {
            this->x/other,
            this->y/other,
            this->z/other
        };
    }

    __device__ __host__ mephvec3 operator-(const mephvec3& other) const
    {
        return {
            this->x-other.x,
            this->y-other.y,
            this->z-other.z
        };
    }
    __device__ __host__ mephvec3 operator+(const mephvec3& other) const
    {
        return {
            this->x+other.x,
            this->y+other.y,
            this->z+other.z
        };
    }
    __device__ __host__ void operator+=(const mephvec3& other)
    {
        this->x+= other.x;
        this->y+= other.y;
        this->z+= other.z;
    }
    __device__ __host__ mephvec3 square()
    {
        return (*this)*(*this);
    }
    [[nodiscard]] __device__ __host__ float sum() const
    {
        return this->x+this->y+this->z;
    }
};
// Particle Struct
struct Particle
{
    float mass;
    mephvec3 pos;
    mephvec3 vel;
    float size;
    float radius;
};
// Variables
float timeStep = 250;
mephvec3 cameraPos{};
float gravitationalConstant =  1.559*pow(10,(-13)); // ly cubed/(solar mass (year) squared)

void initArrays(std::vector<VulkanApp::Vertex> vertecies);
void cleanup();
void copyData(std::vector<VulkanApp::Vertex>* vertecies);

Particle* particleArrayDevice;
Particle* particleArrayHost;
mephvec3* forces;
mephvec3* forces2;
mephvec3* momentumArrayDevice;
float* totalMomentum;
float* momentumBuffer;
float* energyBuffer;
float* energyArrayDevice;
float* totalEnergy;

uint32_t gridD;
uint32_t size;
uint32_t currStep = 0;
bool force1 = true;

// Kernels
__global__ void initForces(mephvec3* force, mephvec3* force2, uint32_t sizef)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=sizef)
    {
        return;
    }
    force[particleId] = {0,0,0};
    force2[particleId] = {0,0,0};
}

__global__ void getAllForces(Particle* particles, mephvec3* forcesf, float g, uint32_t sizef, mephvec3 cameraPosf)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=sizef)
    {
        return;
    }

    float absoluteDistance;
    float totalForce;
    mephvec3 currForce = {0,0,0};
    mephvec3 delta;
    Particle currParticle = particles[particleId];
    for(uint32_t index = 0; index<sizef; index++)
    {
        if (particleId != index)
        {
            delta = currParticle.pos-particles[index].pos;
            absoluteDistance = sqrt(delta.square().sum());
            totalForce = -g * (currParticle.mass * particles[index].mass) /(absoluteDistance*absoluteDistance*absoluteDistance); // Gravity
            totalForce *= (1-exp(-(absoluteDistance-currParticle.radius-particles[index].radius)));
            currForce += delta*totalForce;
        }
    }
    forcesf[particleId] = currForce;
    particles[particleId].size=4/(currParticle.pos-cameraPosf).square().sum();
}

__global__ void getFinalPosition(mephvec3* forcesf, Particle* particles, uint32_t sizef, float timeStepf)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=sizef)
    {
        return;
    }
    float time = timeStepf*timeStepf;
    particles[particleId].pos += particles[particleId].vel*timeStepf+(forcesf[particleId]*time/(particles[particleId].mass*2));
}

__global__ void getFinalVelocity(mephvec3 *forcesf,mephvec3* forcesf2, Particle* particles, uint32_t sizef, float timeStepf)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=sizef)
    {
        return;
    }
    particles[particleId].vel += (forcesf[particleId]+forcesf2[particleId])/(particles[particleId].mass*2)*timeStepf;
}

__global__ void getAllEnergies(Particle* particles, float* energyArray, float g, size_t sizef)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    energyArray[particleId] = 0;
    if(particleId>=sizef)
    {
        return;
    }
    float absoluteDistance;
    float particleEnergy = particles[particleId].mass*particles[particleId].vel.square().sum();
    mephvec3 delta;
    for(uint32_t index = 0; index<sizef; index++)
    {
        if (particleId != index)
        {
            delta = particles[particleId].pos-particles[index].pos;
            absoluteDistance = sqrt(delta.square().sum());
            particleEnergy -=g*particles[particleId].mass*particles[index].mass/absoluteDistance;
        }
    }
    energyArray[particleId] = particleEnergy;
}

__global__ void reduceEnergies(float* indata, float* outdata) {
    __shared__ float temp[THREADS_PER_KERNEL];
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    temp[threadIdx.x] = indata[id];
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
        uint32_t index = 2 * i * threadIdx.x;
        if ((index + i) < blockDim.x) {
            temp[index] += temp[index + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(outdata, temp[0]);
    }
}

__global__ void getAllMomentum(Particle* particles, mephvec3* momentumArray, size_t sizef)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    momentumArray[particleId] = {0,0,0};
    if(particleId>=sizef)
    {
        return;
    }
    momentumArray[particleId] = particles[particleId].vel * particles[particleId].mass;
}

__global__ void reduceMomentum(mephvec3* indata, float* outdata) {
    __shared__ float tempx[THREADS_PER_KERNEL];
    __shared__ float tempy[THREADS_PER_KERNEL];
    __shared__ float tempz[THREADS_PER_KERNEL];
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    tempx[threadIdx.x] = indata[id].x;
    tempy[threadIdx.x] = indata[id].y;
    tempz[threadIdx.x] = indata[id].z;
    __syncthreads();
    for (int i = 1; i < blockDim.x; i *= 2) {
        uint32_t index = 2 * i * threadIdx.x;
        if ((index + i) < blockDim.x) {
            tempx[index] += tempx[index + i];
            tempy[index] += tempy[index + i];
            tempz[index] += tempz[index + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&outdata[0], tempx[0]);
        atomicAdd(&outdata[1], tempy[0]);
        atomicAdd(&outdata[2], tempz[0]);
    }
}
// Memory Functions
void initArrays(std::vector<VulkanApp::Vertex> vertecies)
{
    // Kinematics
    size = vertecies.size();
    gridD = std::ceil(size/(float)(THREADS_PER_KERNEL));
    cudaMalloc(&particleArrayDevice,sizeof(Particle)*size);
    cudaMalloc(&forces,sizeof(mephvec3)*size);
    cudaMalloc(&forces2, sizeof(mephvec3)*size);
    particleArrayHost = static_cast<Particle *>(malloc(sizeof(Particle) * size));
    for(uint32_t i = 0; i<size; i++)
    {
        particleArrayHost[i].pos = vertecies[i].pos;
        particleArrayHost[i].mass = 1;
        particleArrayHost[i].vel = {0,0,0};
        particleArrayHost[i].radius = 0.001;
    }
    particleArrayHost[0].pos = {0,0,0};
    particleArrayHost[0].mass = 3;
    particleArrayHost[0].radius = 0.1;
    particleArrayHost[0].vel= {0,0,0};
    cudaMemcpy(particleArrayDevice,particleArrayHost,size*sizeof(Particle),cudaMemcpyHostToDevice);
    initForces<<<ceil(size/1024.0),1024>>>(forces, forces2, size);
    // Momentum
    cudaMalloc(&momentumBuffer,sizeof(float)*3);
    cudaMalloc(&momentumArrayDevice,sizeof(mephvec3)*std::ceil(size/(float)(THREADS_PER_KERNEL))*THREADS_PER_KERNEL);
    totalMomentum = (float*) malloc(sizeof(float)*3);
    // Energy
    cudaMalloc(&energyBuffer, sizeof(float));
    cudaMalloc(&energyArrayDevice, sizeof(float)*std::ceil(size/(float)(THREADS_PER_KERNEL))*THREADS_PER_KERNEL);
    totalEnergy = (float*) malloc(sizeof(float));
}

void cleanup()
{
    // Kinematics
    free(particleArrayHost);
    cudaFree(particleArrayDevice);
    cudaFree(forces);
    cudaFree(forces2);
    // Momentum
    cudaFree(momentumBuffer);
    cudaFree(momentumArrayDevice);
    free(totalMomentum);
    // Energy
    cudaFree(energyBuffer);
    cudaFree(energyArrayDevice);
    free(totalEnergy);
}


void copyData(std::vector<VulkanApp::Vertex> *vertecies)
{
    std::vector<VulkanApp::Vertex>& vert = *vertecies;
    for(uint32_t i = 0; i<size; i++)
    {
        vert[i].pos = {particleArrayHost[i].pos.x,particleArrayHost[i].pos.y,particleArrayHost[i].pos.z,particleArrayHost[i].size};
    }
}

void sync()
{
    cudaMemcpy(particleArrayHost,particleArrayDevice,size*sizeof(Particle),cudaMemcpyDeviceToHost);
}
// Main functions
void initCuda(VulkanApp* it)
{
    cameraPos = it->eye;
    initArrays(it->vertices);
}

void step(VulkanApp* it)
{
    if(force1)
    {
        getFinalPosition<<<gridD,THREADS_PER_KERNEL>>>(forces2, particleArrayDevice,size,timeStep);
        getAllForces<<<gridD,THREADS_PER_KERNEL>>>(particleArrayDevice,forces,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<gridD,THREADS_PER_KERNEL>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = false;
    }
    else
    {
        getFinalPosition<<<gridD,THREADS_PER_KERNEL>>>(forces, particleArrayDevice,size,timeStep);
        getAllForces<<<gridD,THREADS_PER_KERNEL>>>(particleArrayDevice,forces2,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<gridD,THREADS_PER_KERNEL>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = true;
    }
    if(it->calcEnergy)
    {
        getAllEnergies<<<gridD,THREADS_PER_KERNEL>>>(particleArrayDevice,energyArrayDevice,gravitationalConstant,size);
        cudaMemset (energyBuffer, 0, sizeof(float));
        reduceEnergies<<<gridD,THREADS_PER_KERNEL>>>(energyArrayDevice,energyBuffer);
        cudaMemcpy(totalEnergy,energyBuffer,sizeof(float),cudaMemcpyDeviceToHost);
        *totalEnergy = *totalEnergy/2;
        // std::cout << "Energy "<< currStep << ":" << *totalEnergy << '\n';
        std::cout << currStep << " " << *totalEnergy << '\n';
    }
    if(it->calcMomentum)
    {
        getAllMomentum<<<gridD,THREADS_PER_KERNEL>>>(particleArrayDevice,momentumArrayDevice,size);
        cudaMemset (&momentumBuffer[0], 0, sizeof(float));
        cudaMemset (&momentumBuffer[1], 0, sizeof(float));
        cudaMemset (&momentumBuffer[2], 0, sizeof(float));
        reduceMomentum<<<gridD,THREADS_PER_KERNEL>>>(momentumArrayDevice,momentumBuffer);
        cudaMemcpy(totalMomentum,momentumBuffer,3*sizeof(float),cudaMemcpyDeviceToHost);
        //std::cout << "Momentum "<< currStep << ":(" << totalMomentum[0] << ',' << totalMomentum[1] << ',' << totalMomentum[2] << ")\n";
        std::cerr << currStep << " " << totalMomentum[0] << " " << totalMomentum[1] << " " << totalMomentum[2] << "\n";
    }
    sync();
    copyData(&it->vertices);
    currStep++;
}
void cleanupCuda()
{
    cleanup();
}
// Main
int main(int argc, char** argv)
{
    VulkanApp app;
    app.initCuda = initCuda;
    app.cudaStep = step;
    app.cleanupCuda = cleanupCuda;

    for(int i = 0; i<argc; i++)
    {
        if(strcmp(argv[i],"-d=1")==0)
        {
            app.enableValidationLayers = true;
        }
        else if(strcmp(argv[i],"-d=2")==0)
        {
            app.enableValidationLayers = true;
            app.printDebugMessages = true;
        }
        else if(strcmp(argv[i],"-interOp")==0)
        {
            app.interOp = true;
        }
        else if(strcmp(argv[i],"-e=1")==0)
        {
            app.calcEnergy = true;
        }
        else if(strcmp(argv[i],"-e=2")==0)
        {
            app.calcEnergy = true;
            app.exportEnergy = true;
        }
        else if(strcmp(argv[i],"-m=1")==0)
        {
            app.calcMomentum = true;
        }
        else if(strcmp(argv[i],"-m=2")==0)
        {
            app.calcMomentum = true;
            app.exportMomentum = true;
        }
        else if(strcmp(argv[i],"--single-frame")==0)
        {
            app.singleFrame = true;
        }
        else if(strcmp(argv[i],"--vsync-off")==0)
        {
            app.vsyncOff = true;
        }
        else if(strlen(argv[i])>9 && strcmp(std::string(argv[i]).substr(0,9).c_str(),"--points=")==0)
        {
            app.pointCount = std::stoi(std::string(argv[i]).substr(9,std::string::npos));
        }
    }

    if(app.enableValidationLayers)
    {
        std::cout << "Validation Layers Enabled: ";
        for (auto validationLayer : app.validationLayers)
        {
            std::cout << validationLayer << " ";
        }
        std::cout << std::endl;
    }
    try
    {
        app.run();
    }
    catch (const std::exception & e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}