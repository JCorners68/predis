# Predis Docker Development Environment

This document describes the containerized development environment for Predis.

## Quick Start

```bash
# Start development environment
./dev_scripts/start_dev.sh

# Enter development container
docker-compose -f docker-compose.dev.yml exec predis-dev bash

# Stop environment
./dev_scripts/stop_dev.sh
```

## Container Details

### Base Image
- `nvidia/cuda:12.8-devel-ubuntu22.04` with GPU development support
- Non-root user `developer` (UID 1000) for security
- Development tools: gcc-11, cmake, gdb, valgrind

### Ports
- `6379`: Predis cache service
- `6380`: Redis benchmark service  
- `8080`: API and monitoring endpoint
- `9090`: Development tools
- `9091`: Prometheus monitoring (when enabled)

### Volumes
- `.:/workspace:cached` - Source code (live editing)
- `predis-build:/workspace/build` - Build artifacts (persistent)
- `predis-logs:/workspace/logs` - Application logs

## Usage

### Development Workflow
```bash
# Build the project
cd /workspace
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run tests
make test

# Check GPU access
nvidia-smi
```

### Benchmarking
```bash
# Start Redis for comparison
docker-compose -f docker-compose.dev.yml --profile benchmark up -d redis-benchmark

# Redis will be available on port 6380
redis-cli -p 6380 ping
```

### Monitoring
```bash
# Start Prometheus monitoring
docker-compose -f docker-compose.dev.yml --profile monitoring up -d monitoring

# Access Prometheus at http://localhost:9091
```

## GPU Requirements

- NVIDIA GPU with CUDA 12.8+ support
- nvidia-docker2 installed on host
- GPU drivers compatible with CUDA 12.8

### GPU Troubleshooting
```bash
# Test GPU access in container
docker-compose -f docker-compose.dev.yml exec predis-dev nvidia-smi

# Check GPU memory
docker-compose -f docker-compose.dev.yml exec predis-dev nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Development Environment Features

### Shell Aliases
- `ll` - ls -la
- `la` - ls -A  
- `l` - ls -CF

### Installed Tools
- Build: cmake, gcc-11, g++-11, clang-14
- Debug: gdb, valgrind
- Python: python3, pip3 with ML dependencies
- Utilities: vim, htop, tree, curl, wget

## Profiles

Use Docker Compose profiles to start different service combinations:

```bash
# Development only
docker-compose -f docker-compose.dev.yml up -d

# With Redis benchmarking
docker-compose -f docker-compose.dev.yml --profile benchmark up -d

# With monitoring
docker-compose -f docker-compose.dev.yml --profile monitoring up -d

# Everything
docker-compose -f docker-compose.dev.yml --profile benchmark --profile monitoring up -d
```

## Common Issues

### Permission Issues
The container runs as `developer` user (UID 1000). If you encounter permission issues with mounted files, ensure your host user has UID 1000 or adjust the container's USER_UID build arg.

### GPU Access
If GPU access fails:
1. Verify nvidia-docker2 is installed: `docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi`
2. Check CUDA version compatibility
3. Restart Docker daemon after nvidia-docker2 installation

### Build Issues
- Clean build directory: `rm -rf build/*`
- Update dependencies: `pip3 install -r requirements.txt`
- Check CMake version: `cmake --version`