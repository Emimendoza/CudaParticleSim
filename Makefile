CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
NVCC=/usr/local/cuda/bin/nvcc

VulkanTest: main.cu
	@$(NVCC) $(CFLAGS) -o VulkanTest main.cu $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	@./VulkanTest -d=2

clean:
	@rm -f VulkanTest

all: VulkanTest test clean