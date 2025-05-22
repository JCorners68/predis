#!/bin/bash

# Install Development Tools for Predis
# Run this script to install all required development tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

print_status "Installing development tools for Predis..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root!"
    exit 1
fi

# Update package list
print_status "Updating package list..."
sudo apt update

# Install C++ development tools
print_status "Installing C++ development tools..."
sudo apt install -y \
    build-essential \
    cmake \
    clang-format \
    clang-tidy \
    valgrind \
    gdb \
    pkg-config \
    libssl-dev \
    libffi-dev

# Install Python development tools
print_status "Installing Python development tools..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv

# Install additional tools
print_status "Installing additional development tools..."
sudo apt install -y \
    git \
    curl \
    wget \
    unzip \
    htop \
    tree

# Install pre-commit (Python package)
print_status "Installing pre-commit..."
pip3 install --user pre-commit

# Install cmake-format
print_status "Installing cmake-format..."
pip3 install --user cmake-format

# Install Google Test (optional)
print_status "Installing Google Test development files..."
sudo apt install -y libgtest-dev

print_success "All development tools installed successfully!"

print_status "Verifying installation..."
echo "CMake: $(cmake --version | head -1)"
echo "Clang-format: $(clang-format --version)"
echo "GCC: $(gcc --version | head -1)"
echo "Python: $(python3 --version)"

print_success "Development environment ready!"
print_status "You can now run: make format, make lint, make build"