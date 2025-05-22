#!/bin/bash

# Predis Development Environment Startup Script

set -e

echo "🚀 Starting Predis Development Environment..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "⚠️  Warning: GPU access may not work. Make sure nvidia-docker2 is installed."
fi

# Build development container if needed
echo "🔨 Building development container..."
docker-compose -f docker-compose.dev.yml build predis-dev

# Start the development environment
echo "🐳 Starting containers..."
docker-compose -f docker-compose.dev.yml up -d predis-dev

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 3

# Test GPU access
echo "🔍 Testing GPU access..."
if docker-compose -f docker-compose.dev.yml exec predis-dev nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU access confirmed"
    docker-compose -f docker-compose.dev.yml exec predis-dev nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "❌ GPU access failed - development will continue without GPU"
fi

# Display container information
echo ""
echo "📋 Development Environment Ready!"
echo "Container: predis-dev"
echo "Workspace: /workspace"
echo "Ports:"
echo "  - 6379: Predis cache"
echo "  - 8080: API/monitoring"
echo "  - 9090: Development tools"
echo ""
echo "🔧 Available commands:"
echo "  docker-compose -f docker-compose.dev.yml exec predis-dev bash  # Enter container"
echo "  docker-compose -f docker-compose.dev.yml logs -f predis-dev    # View logs"
echo "  docker-compose -f docker-compose.dev.yml down                  # Stop environment"
echo ""
echo "🧪 To start Redis for benchmarking:"
echo "  docker-compose -f docker-compose.dev.yml --profile benchmark up -d redis-benchmark"
echo ""

# Optionally enter the container
read -p "Enter development container now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.dev.yml exec predis-dev bash
fi