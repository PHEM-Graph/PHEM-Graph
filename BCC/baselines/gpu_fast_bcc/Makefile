# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 --expt-relaxed-constexpr
ARCH = -arch=sm_89
INCLUDES = -I./include

# Source and object directories
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include

# Source files and corresponding object files
SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC))

# Executable name
EXEC = main

# Default target
all: $(EXEC)

# Create object directory if it doesn't exist, and compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) $(ARCH) $(INCLUDES) -c $< -o $@

# Link object files to create the executable
$(EXEC): $(OBJ)
	$(NVCC) $(CXXFLAGS) $(ARCH) -o $@ $^

# Ensure object directory exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up object files and the executable
clean:
	rm -rf $(OBJ_DIR) $(EXEC)

# Phony targets
.PHONY: all clean
