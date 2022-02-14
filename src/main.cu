#include <iostream>
#include "VulkanApp.h"
#include "CudaGravSim.h"

CudaGravSim *sim;

void initCuda(VulkanApp* it)
{
    sim = new CudaGravSim(it->eye);
    sim->initArrays(it->vertices);
}

void step(VulkanApp* it)
{
    sim->step();
    sim->copyData(&it->vertices);
}
void cleanupCuda()
{
    sim->cleanup();
    delete sim;
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