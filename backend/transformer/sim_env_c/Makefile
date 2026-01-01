# Makefile for CTF Simulator C Library

# Detect OS
UNAME_S := $(shell uname -s)

# Compiler settings
CC = gcc
CFLAGS = -O3 -Wall -Wextra -fPIC -std=c99

# Output settings
ifeq ($(UNAME_S),Darwin)
    # macOS
    SHARED_EXT = dylib
    SHARED_FLAGS = -dynamiclib
else
    # Linux
    SHARED_EXT = so
    SHARED_FLAGS = -shared
endif

# Directories
BUILD_DIR = build
SRC_DIR = .

# Files
SOURCES = ctf_sim.c cJSON.c
OBJECTS = $(SOURCES:%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/libctfsim.$(SHARED_EXT)

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link shared library
$(TARGET): $(OBJECTS)
	$(CC) $(SHARED_FLAGS) -o $@ $(OBJECTS)
	@echo "Built: $(TARGET)"

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Test compilation
test: $(TARGET)
	@echo "Library built successfully!"
	@ls -la $(TARGET)

.PHONY: all clean test
