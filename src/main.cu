#include <iostream>
#include "VulkanApp.h"

struct mephvec3{

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

    __device__ __host__ mephvec3& operator=(glm::vec4& vec1)
    {
        this->x = vec1.x;
        this->y = vec1.y;
        this->z = vec1.z;
        return *this;
    }

    __device__ __host__ mephvec3 operator*(mephvec3 other) const
    {
        mephvec3 result{};
        result.x=other.x*this->x;
        result.y=other.y*this->y;
        result.z=other.z*this->z;
        return result;
    }
    __device__ __host__ mephvec3 operator*(float other) const
    {
        mephvec3 result{};
        result.x = other * this->x;
        result.y = other * this->y;
        result.z = other * this->z;
        return result;
    }

    __device__ __host__ mephvec3 operator/(float other) const
    {
        mephvec3 results{};
        results.x = this->x/other;
        results.y = this->y/other;
        results.z = this->z/other;
        return results;
    }

    __device__ __host__ mephvec3 operator-(mephvec3 other) const
    {
        mephvec3 result{};
        result.x = this->x-other.x;
        result.y = this->y-other.y;
        result.z = this->z-other.z;
        return result;
    }
    __device__ __host__ mephvec3 operator+(mephvec3 other) const
    {
        mephvec3 result{};
        result.x = this->x+other.x;
        result.y = this->y+other.y;
        result.z = this->z+other.z;
        return result;
    }
    __device__ __host__ void operator+=(mephvec3 other)
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
struct Particle
{
    float mass;
    mephvec3 pos;
    mephvec3 vel;
    float size;
};

float timeStep = 1000;
mephvec3 cameraPos{};
float gravitationalConstant =  1.559*pow(10,(-13)); // ly cubed/(solar mass (year) squared)

void initArrays(std::vector<VulkanApp::Vertex> vertecies);
void step();
void cleanup();
void copyData(std::vector<VulkanApp::Vertex>* vertecies);

Particle* particleArrayDevice;
Particle* particleArrayHost;
mephvec3* forces;
mephvec3* forces2;
uint32_t size;
bool force1 = true;

void sync();

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
    forcesf[particleId] = {0,0,0};
    for(uint32_t index = 0; index<sizef; index++)
    {
        if (particleId != index)
        {
            mephvec3 delta = particles[particleId].pos - particles[index].pos;

            float absoluteDistance = sqrt(delta.square().sum());
            float totalForce = g * (particles[particleId].mass * particles[index].mass) /pow(absoluteDistance,3);

            forcesf[index] += delta*totalForce;
        }
    }
    particles[particleId].size=4/(particles[particleId].pos-cameraPosf).square().sum();
}

__global__ void getFinalPosition(mephvec3* forcesf, Particle* particles, uint32_t sizef, float timeStepf)
{
    uint32_t particleId = blockDim.x * blockIdx.x + threadIdx.x;
    if(particleId>=sizef)
    {
        return;
    }
    float time = pow(timeStepf,2);
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

void initArrays(std::vector<VulkanApp::Vertex> vertecies)
{
    size = vertecies.size();
    cudaMalloc(&particleArrayDevice,sizeof(Particle)*size);
    cudaMalloc(&forces,sizeof(mephvec3)*size);
    cudaMalloc(&forces2, sizeof(mephvec3)*size);
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

void cleanup()
{
    free(particleArrayHost);
    cudaFree(particleArrayDevice);
    cudaFree(forces);
    cudaFree(forces2);
}

void step()
{
    if(force1)
    {
        getFinalPosition<<<std::ceil(size/512.0),512>>>(forces2, particleArrayDevice,size,timeStep);
        getAllForces<<<std::ceil(size/512.0),512>>>(particleArrayDevice,forces,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<std::ceil(size/512.0),512>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = false;
    }
    else
    {
        getFinalPosition<<<std::ceil(size/512.0),512>>>(forces, particleArrayDevice,size,timeStep);
        getAllForces<<<std::ceil(size/512.0),512>>>(particleArrayDevice,forces2,gravitationalConstant,size,cameraPos);
        getFinalVelocity<<<std::ceil(size/512.0),512>>>(forces,forces2,particleArrayDevice,size,timeStep);
        force1 = true;
    }
    sync();
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

void initCuda(VulkanApp* it)
{
    cameraPos = it->eye;
    initArrays(it->vertices);
}

void step(VulkanApp* it)
{
    step();
    copyData(&it->vertices);
}
void cleanupCuda()
{
    cleanup();
}

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
        if(strcmp(argv[i],"-d=2")==0)
        {
            app.enableValidationLayers = true;
            app.printDebugMessages = true;
        }
        if(strcmp(argv[i],"--single-frame")==0)
        {
            app.singleFrame = true;
        }
        if(strlen(argv[i])>9 && strcmp(std::string(argv[i]).substr(0,9).c_str(),"--points=")==0)
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