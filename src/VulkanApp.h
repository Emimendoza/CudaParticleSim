// VulkanApp.h
#ifndef TEST_VULKANAPP_H
#define TEST_VULKANAPP_H
#define GLFW_INCLUDE_VULKAN
#include <optional>
#include <GLFW/glfw3.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <array>

class VulkanApp
{
public:
    struct Vertex
    {
        glm::vec4 pos;
        glm::vec3 color;
        static VkVertexInputBindingDescription getBindingDescription();
        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
    };

    void (*cudaStep)(VulkanApp*);
    void (*initCuda)(VulkanApp*);
    void (*cleanupCuda)();

    uint32_t pointCount = 1000;
    bool singleFrame = false;
    bool vsyncOff = false;
    bool calcEnergy = false;
    bool exportEnergy = false;
    bool calcMomentum = false;
    bool exportMomentum = false;
    bool interOp = false;
    const uint32_t WIDTH = 800, HEIGHT = 600;
    const uint MAX_FRAMES_IN_FLIGHT = 2;

    const std::vector<const char*> validationLayers =
            {
                    "VK_LAYER_KHRONOS_validation"
            };

    std::vector<const char*> deviceExtensions =
            {
                    VK_KHR_SWAPCHAIN_EXTENSION_NAME
            };

    glm::vec3 eye = {1, 0, 1};

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    bool enableValidationLayers = false;
    bool printDebugMessages = false;

    void run();

    std::vector<Vertex> vertices;

    std::vector<uint32_t> indices;

private:
    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete();
    };
    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice vkDevice;
    VkQueue graphicsQueue;
    VkSurfaceKHR surface;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer copyVertexBuffer;
    VkDeviceMemory copyVertexBufferMemory;
    VkDeviceSize copyVertexBufferSize;
    void* localCopyVertexBufferMemory;
    int externMemoryHandle;
    VkDescriptorSetLayout descriptorSetLayout;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    VkDescriptorPool descriptorPool;
    bool framebufferResized = false;
    std::vector<VkDescriptorSet> descriptorSets;

    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

    void createCopyBuffer();
    void copyVertexToBuffer();
    static float randNum(double low, double high);
    void createPoints();
    void createDescriptorSets();
    void createDescriptorPool();
    void updateUniformBuffer(uint32_t currentImage);
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void recreateSwapChain();
    void createUniformBuffers();
    void createDescriptorSetLayout();
    void createIndexBuffer();
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool external);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void cleanupSwapChain();
    void createVertexBuffer();
    void createSyncObjects();
    void drawFrame();
    void createCommandBuffers();
    void createCommandPool();
    void createFrameBuffers();
    void createRenderPass();
    void createImageViews();
    void createSwapChain();
    void createSurface();
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent (const VkSurfaceCapabilitiesKHR& capabilities);
    void createLogicalDevice();
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    void createInstance();
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void setupDebugMessenger();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
    void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData)
    {
        if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            // Message is important enough to show
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        }
        return VK_FALSE;
    }
    void createGraphicsPipeline();
    static std::vector<char> readFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);

};


#endif //TEST_VULKANAPP_H
