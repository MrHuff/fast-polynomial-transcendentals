# --- CUDA Benchmark Makefile ---

# Compiler
NVCC = nvcc

# Flags: -O3 for optimization, -I. to find headers in this folder
# Adjust -arch=sm_70 to your GPU's compute capability if needed
NVCCFLAGS = -std=c++17 -O3 -arch=sm_100 -I.

# --- Files ---
# This is the executable we will create
TARGET = cuda_benchmark

# THIS LINE IS FIXED:
# It now matches your filename 'benchmark.cu'
SOURCE = benchmark.cu


# 'make' or 'make all' will build the target
all: $(TARGET) benchmark_backward

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)
	@echo "CUDA benchmark compiled: ./$(TARGET)"

benchmark_backward: benchmark_backward.cu
	$(NVCC) $(NVCCFLAGS) -o benchmark_backward benchmark_backward.cu
	@echo "CUDA benchmark_backward compiled: ./benchmark_backward"

# 'make run' will build and run
run: all
	@./$(TARGET)
	@./benchmark_backward

# 'make clean' removes the executable
clean:
	@rm -f $(TARGET) benchmark_backward
	@echo "Cleaned CUDA benchmark"

.PHONY: all run clean benchmark_backward