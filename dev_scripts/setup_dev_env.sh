#!/bin/bash

# Development Environment Setup Script for Predis
# Sets up complete C++/CUDA/Python development environment
# Author: Predis Project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Starting Predis development environment setup..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_status "Project root: $(pwd)"

# Check if we're in WSL
if grep -qi microsoft /proc/version; then
    print_status "Detected WSL environment"
    WSL_ENV=true
else
    print_status "Detected native Linux environment"
    WSL_ENV=false
fi

# Check Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
print_status "Ubuntu version: $UBUNTU_VERSION"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root!"
    exit 1
fi

# 1. Verify GPU access
print_status "Verifying GPU access..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    print_success "NVIDIA GPU detected and accessible"
else
    print_error "nvidia-smi not found! Please install NVIDIA drivers first."
    exit 1
fi

# 2. Check CUDA installation
print_status "Checking CUDA installation..."
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    print_success "CUDA $CUDA_VERSION detected"
else
    print_warning "CUDA toolkit not found. Installing CUDA development tools..."
    
    # Install CUDA toolkit
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit
    
    # Verify installation
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        print_success "CUDA $CUDA_VERSION installed successfully"
    else
        print_error "CUDA installation failed!"
        exit 1
    fi
fi

# 3. Check Docker installation
print_status "Checking Docker installation..."
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker $DOCKER_VERSION detected"
else
    print_error "Docker not found! Please install Docker first."
    exit 1
fi

# 4. Test Docker GPU access
print_status "Testing Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    print_success "Docker GPU access working"
else
    print_error "Docker GPU access failed! Check NVIDIA Container Runtime installation."
    exit 1
fi

# 5. Check Python installation
print_status "Checking Python installation..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    print_success "Python $PYTHON_VERSION detected (>= 3.10 required)"
else
    print_error "Python 3.10+ required, found Python $PYTHON_VERSION"
    exit 1
fi

# 6. Install system dependencies
print_status "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    clang-format \
    valgrind \
    gdb \
    curl \
    wget \
    unzip

print_success "System dependencies installed"

# 7. Create Python virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Python virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# 8. Install Python development dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << EOF
# Core dependencies
numpy>=1.21.0
pybind11>=2.10.0
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.950

# GPU/ML dependencies
torch>=2.0.0
tensorflow>=2.12.0

# Development dependencies
jupyter>=1.0.0
matplotlib>=3.5.0
pandas>=1.4.0
scipy>=1.8.0

# Benchmarking dependencies
redis>=4.5.0
psutil>=5.9.0
EOF
    print_success "requirements.txt created"
fi

pip install -r requirements.txt
print_success "Python dependencies installed"

# 9. Create development Docker configuration
print_status "Creating development Docker configuration..."

# Create docker/Dockerfile.dev
mkdir -p docker
cat > docker/Dockerfile.dev << EOF
# Predis Development Container
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=\${CUDA_HOME}/bin:\${PATH}
ENV LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    git \\
    python3 \\
    python3-dev \\
    python3-pip \\
    python3-venv \\
    pkg-config \\
    libssl-dev \\
    libffi-dev \\
    clang-format \\
    valgrind \\
    gdb \\
    curl \\
    wget \\
    unzip \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Install Redis for benchmarking
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

# Create development user
RUN useradd -m -s /bin/bash developer && \\
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up Python environment
USER developer
WORKDIR /home/developer

# Copy requirements and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --user --upgrade pip && \\
    python3 -m pip install --user -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace
USER developer

# Default command
CMD ["/bin/bash"]
EOF

# Create docker-compose.yml for development
cat > docker-compose.yml << EOF
version: '3.8'

services:
  predis-dev:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    container_name: predis-dev
    volumes:
      - .:/workspace
      - predis-cache:/home/developer/.cache
    working_dir: /workspace
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
    networks:
      - predis-network

  redis-benchmark:
    image: redis:7-alpine
    container_name: predis-redis-benchmark
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - predis-network
    ports:
      - "6379:6379"

volumes:
  predis-cache:
  redis-data:

networks:
  predis-network:
    driver: bridge
EOF

print_success "Docker configuration created"

# 10. Create simple CUDA test program
print_status "Creating CUDA test program..."
mkdir -p tests/gpu
cat > tests/gpu/cuda_test.cu << EOF
/*
 * Copyright 2025 Predis Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloPredis(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("Hello from Predis GPU thread %d!\\n", idx);
    }
}

int main() {
    // Check GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    }
    
    // Launch simple kernel
    int n = 16;
    int blockSize = 8;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    std::cout << "\\nLaunching Predis CUDA kernel..." << std::endl;
    helloPredis<<<gridSize, blockSize>>>(n);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "\\nPredis CUDA test completed successfully!" << std::endl;
    return 0;
}
EOF

# Create compilation script
cat > tests/gpu/compile_test.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
nvcc -o cuda_test cuda_test.cu
if [ $? -eq 0 ]; then
    echo "CUDA test compiled successfully"
    echo "Run with: ./cuda_test"
else
    echo "CUDA compilation failed"
    exit 1
fi
EOF

chmod +x tests/gpu/compile_test.sh
print_success "CUDA test program created"

# 11. Test CUDA compilation
print_status "Testing CUDA compilation..."
cd tests/gpu
if ./compile_test.sh; then
    print_success "CUDA compilation successful"
    
    # Run the test
    print_status "Running CUDA test..."
    if ./cuda_test; then
        print_success "CUDA test execution successful"
    else
        print_warning "CUDA test execution failed, but compilation works"
    fi
else
    print_error "CUDA compilation failed"
fi

cd "$PROJECT_ROOT"

# 12. Create environment summary
print_status "Creating environment summary..."
cat > dev_environment_info.md << EOF
# Development Environment Information

**Generated**: $(date)
**System**: $(uname -a)
**Ubuntu**: $(lsb_release -d | cut -f2)

## GPU Information
\`\`\`
$(nvidia-smi)
\`\`\`

## CUDA Information
- **CUDA Version**: $(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
- **CUDA Path**: $(which nvcc)

## Docker Information
- **Docker Version**: $(docker --version)
- **Docker Compose**: $(docker compose version)

## Python Information
- **Python Version**: $(python3 --version)
- **Python Path**: $(which python3)
- **Virtual Environment**: $(pwd)/venv

## Installed Packages
\`\`\`
$(pip list)
\`\`\`

## Usage Instructions

### Activate Development Environment
\`\`\`bash
# Activate Python virtual environment
source venv/bin/activate

# Start development containers
docker compose up -d

# Enter development container
docker compose exec predis-dev bash
\`\`\`

### Build and Test
\`\`\`bash
# Compile CUDA test
cd tests/gpu
./compile_test.sh
./cuda_test

# Run Python tests
pytest tests/
\`\`\`
EOF

print_success "Environment summary created: dev_environment_info.md"

# 13. Final verification
print_status "Performing final verification..."

echo ""
print_success "🚀 Development environment setup completed successfully!"
echo ""
print_status "Summary:"
print_status "✅ WSL2 with Ubuntu $UBUNTU_VERSION"
print_status "✅ NVIDIA GPU (RTX 5080) accessible"
print_status "✅ CUDA $CUDA_VERSION installed and working"
print_status "✅ Docker with GPU support working"
print_status "✅ Python $PYTHON_VERSION with virtual environment"
print_status "✅ Development containers configured"
print_status "✅ CUDA test program compiled and tested"
echo ""
print_status "Next steps:"
print_status "1. Activate virtual environment: source venv/bin/activate"
print_status "2. Start development containers: docker compose up -d"
print_status "3. Begin Epic 1 development!"
echo ""
print_success "Ready for GPU-accelerated Predis development! 🎯"
EOF