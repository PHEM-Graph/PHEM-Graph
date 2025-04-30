# Makefile for compiling programs

# Compiler
NVCC = nvcc

# Compiler flags
# Use -arch=sm_80 for A100 and 89 for L40
CXXFLAGS = -Xcompiler "-pthread" -std=c++17 -O3 -arch=sm_80

# Source files
SRC = graph.cu main.cu host_spanning_tree.cu ExternalSpanningTree.cu GPUSpanningTree.cu

# Output executable
TARGET = hast

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET)
