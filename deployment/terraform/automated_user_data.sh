#!/bin/bash
# Automated Predis development environment setup
set -e

# Logging
exec > >(tee /var/log/user-data.log) 2>&1
echo "=== Predis Environment Setup Started at $(date) ==="

# Update system
echo "=== Updating system ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y

# Install NVIDIA drivers and CUDA
echo "=== Installing NVIDIA drivers ==="
apt-get install -y nvidia-driver-535
echo "NVIDIA driver installed - will be active after reboot"

# Install development tools
echo "=== Installing development tools ==="
apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    python3-dev \
    docker.io \
    curl \
    wget \
    htop \
    vim \
    unzip \
    software-properties-common

# Configure Docker
echo "=== Configuring Docker ==="
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# Install NVIDIA Container Toolkit for Docker GPU support
echo "=== Installing NVIDIA Docker support ==="
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# Install Python packages for ML development
echo "=== Installing Python packages ==="
sudo -u ubuntu python3 -m pip install --user --upgrade pip
sudo -u ubuntu python3 -m pip install --user \
    torch \
    torchvision \
    torchaudio \
    numpy \
    pandas \
    matplotlib \
    jupyter \
    redis \
    fastapi \
    uvicorn \
    pytest \
    psutil \
    pynvml

# Create development directories
echo "=== Setting up development environment ==="
sudo -u ubuntu mkdir -p /home/ubuntu/{projects,scripts,data}

# Create useful scripts
echo "=== Creating utility scripts ==="
cat > /home/ubuntu/scripts/check_gpu.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys

def check_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("NVIDIA-SMI Output:")
        print(result.stdout)
        return result.returncode == 0
    except FileNotFoundError:
        print("nvidia-smi not found")
        return False

def check_pytorch_cuda():
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            # Test GPU computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("GPU computation test: PASSED")
            return True
        return False
    except ImportError:
        print("PyTorch not installed")
        return False
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== GPU Environment Check ===")
    gpu_ok = check_nvidia_smi()
    pytorch_ok = check_pytorch_cuda()
    
    if gpu_ok and pytorch_ok:
        print("\n✅ GPU environment is ready!")
        sys.exit(0)
    else:
        print("\n❌ GPU environment has issues")
        sys.exit(1)
EOF

cat > /home/ubuntu/scripts/predis_dev_status.sh << 'EOF'
#!/bin/bash
echo "=== Predis Development Environment Status ==="
echo "Date: $(date)"
echo ""

echo "🖥️  System Info:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  Kernel: $(uname -r)"
echo "  Uptime: $(uptime -p)"
echo ""

echo "🎮 GPU Info:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read name memory_total memory_used temp; do
        echo "  GPU: $name"
        echo "  VRAM: ${memory_used}MB / ${memory_total}MB"
        echo "  Temp: ${temp}°C"
    done
else
    echo "  ❌ NVIDIA driver not loaded (reboot required)"
fi
echo ""

echo "🐳 Docker Info:"
if systemctl is-active --quiet docker; then
    echo "  Status: ✅ Running"
    echo "  Version: $(docker --version | cut -d' ' -f3 | sed 's/,//')"
else
    echo "  Status: ❌ Not running"
fi
echo ""

echo "🐍 Python Info:"
echo "  Version: $(python3 --version | cut -d' ' -f2)"
echo "  Packages: $(python3 -m pip list --user | wc -l) installed"
echo ""

echo "💾 Storage Info:"
df -h / | tail -1 | awk '{print "  Root: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""

echo "🌐 Network Info:"
echo "  Private IP: $(hostname -I | awk '{print $1}')"
if command -v curl >/dev/null 2>&1; then
    echo "  Public IP: $(curl -s https://ipinfo.io/ip 2>/dev/null || echo 'Unable to detect')"
fi
EOF

# Make scripts executable
chmod +x /home/ubuntu/scripts/*.py
chmod +x /home/ubuntu/scripts/*.sh
chown -R ubuntu:ubuntu /home/ubuntu/scripts

# Add useful aliases to bashrc
cat >> /home/ubuntu/.bashrc << 'EOF'

# Predis development aliases
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias checkgpu='python3 ~/scripts/check_gpu.py'
alias status='~/scripts/predis_dev_status.sh'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

echo "🚀 Predis development environment loaded!"
echo "Commands: gpu, gpuwatch, checkgpu, status"
echo "Run 'status' to see environment info"
EOF

# Create a setup completion indicator
cat > /home/ubuntu/setup_status.txt << 'EOF'
🚀 Predis Development Environment Setup Complete!

✅ NVIDIA drivers installed (reboot required to activate)
✅ Docker installed with GPU support
✅ Python ML packages installed
✅ Development tools ready
✅ Utility scripts created

Next Steps:
1. Reboot the instance: sudo reboot
2. Reconnect via SSH
3. Test GPU: checkgpu
4. Check status: status

Useful Commands:
- gpu          : Show GPU status
- gpuwatch     : Monitor GPU in real-time  
- checkgpu     : Test GPU functionality
- status       : Show environment status
- docker run --gpus all nvidia/cuda:12.2-runtime-ubuntu22.04 nvidia-smi

Development ready for Predis GPU-accelerated caching project!
EOF

chown ubuntu:ubuntu /home/ubuntu/setup_status.txt

# Schedule automatic reboot in 2 minutes to load NVIDIA driver
echo "=== Scheduling reboot to activate NVIDIA driver ==="
echo "Setup complete! Rebooting in 2 minutes to activate GPU driver..."
shutdown -r +2 "Rebooting to activate NVIDIA driver - setup complete"

echo "=== Setup script completed at $(date) ==="