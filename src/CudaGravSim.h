#ifndef TEST_CUDAGRAVSIM_H
#define TEST_CUDAGRAVSIM_H
#include "VulkanApp.h"
#include <vector>
#include <cstdio>

class CudaGravSim
{
public:
    explicit CudaGravSim(glm::vec3 &cameraPos);

    struct vec3{

        double x;
        double y;
        double z;

        __device__ __host__ vec3& operator=(glm::vec3& vec1)
        {
            this->x = vec1.x;
            this->y = vec1.y;
            this->z = vec1.z;
            return *this;
        }

        __device__ __host__ vec3& operator=(glm::vec4& vec1)
        {
            this->x = vec1.x;
            this->y = vec1.y;
            this->z = vec1.z;
            return *this;
        }

        __device__ __host__ vec3 operator*(vec3 other) const
        {
            vec3 result{};
            result.x=other.x*this->x;
            result.y=other.y*this->y;
            result.z=other.z*this->z;
            return result;
        }
        __device__ __host__ vec3 operator*(double other) const
        {
            vec3 result{};
            result.x = other * this->x;
            result.y = other * this->y;
            result.z = other * this->z;
            return result;
        }

        __device__ __host__ vec3 operator/(double other) const
        {
            vec3 results{};
            results.x = this->x/other;
            results.y = this->y/other;
            results.z = this->z/other;
            return results;
        }

        __device__ __host__ vec3 operator-(vec3 other) const
        {
            vec3 result{};
            result.x = this->x-other.x;
            result.y = this->y-other.y;
            result.z = this->z-other.z;
            return result;
        }
        __device__ __host__ vec3 operator+(vec3 other) const
        {
            vec3 result{};
            result.x = this->x+other.x;
            result.y = this->y+other.y;
            result.z = this->z+other.z;
            return result;
        }
        __device__ __host__ void operator+=(vec3 other)
        {
            this->x+= other.x;
            this->y+= other.y;
            this->z+= other.z;
        }
        __device__ __host__ vec3 square()
        {
            return (*this)*(*this);
        }
        __device__ __host__ double sum() const
        {
            return this->x+this->y+this->z;
        }
    };
    struct Particle
    {
        double mass;
        vec3 pos;
        vec3 vel;
        double size;
    };

    double timeStep = 1000;
    vec3 cameraPos{};
    double gravitationalConstant =  1.559*pow(10,(-13)); // ly cubed/(solar mass (year) squared)

    void initArrays(std::vector<VulkanApp::Vertex> vertecies);
    void step();
    void cleanup();
    void copyData(std::vector<VulkanApp::Vertex>* vertecies);
private:

    Particle* particleArrayDevice;
    Particle* particleArrayHost;
    vec3* forces;
    vec3* forces2;
    uint32_t size;
    bool force1 = true;

    void sync();

};

#endif
