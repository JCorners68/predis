#!/bin/bash
# deploy_predis.sh - Complete deployment of Predis to AWS Tesla T4
set -e

# Configuration
AWS_HOST="ubuntu@52.71.97.28"
SSH_KEY="$HOME/.ssh/id_rsa"
REMOTE_DIR="~/predis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Predis Deployment to AWS Tesla T4"
echo "Target: $AWS_HOST"
echo "=================================="

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found at $SSH_KEY${NC}"
    exit 1
fi

# Function to upload with progress
upload_with_progress() {
    local source=$1
    local dest=$2
    local description=$3
    
    echo "📁 Uploading $description..."
    rsync -av --progress --exclude='.git' --exclude='.terraform' \
          --exclude='*.pdf' --exclude='patent*' --exclude='node_modules' \
          --exclude='benchmark_results' --exclude='build' \
          "$source" "$AWS_HOST:$dest"
    echo "✅ $description uploaded successfully"
    echo ""
}

# Step 1: Check system dependencies on remote host
echo "1️⃣ Checking system dependencies..."
ssh -i "$SSH_KEY" "$AWS_HOST" << 'EOF'
echo "📋 Checking required packages..."
required_packages=("cmake" "g++" "python3-pip" "python3-dev" "libboost-all-dev")
missing_packages=()

for pkg in "${required_packages[@]}"; do
    if ! dpkg -l | grep -q "^ii  $pkg"; then
        missing_packages+=("$pkg")
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "⚠️  Missing packages: ${missing_packages[*]}"
    echo "Installing missing packages..."
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
else
    echo "✅ All required packages are installed"
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA toolkit not found. GPU functionality may be limited."
else
    echo "✅ CUDA toolkit found: $(nvcc --version | grep release)"
fi

# Check Python version
python3 --version
EOF
echo ""

# Step 2: Create remote directory structure
echo "2️⃣ Creating remote directory structure..."
ssh -i "$SSH_KEY" "$AWS_HOST" << EOF
mkdir -p $REMOTE_DIR/{build,logs,data,models}
mkdir -p $REMOTE_DIR/deployment/config
echo "✅ Directory structure created:"
ls -la $REMOTE_DIR/
EOF
echo ""

# Step 3: Upload source code
upload_with_progress "src/" "$REMOTE_DIR/" "Source Code (src/)"

# Step 4: Upload ML and data directories
echo "4️⃣ Uploading ML models and data..."
if [ -d "data" ]; then
    upload_with_progress "data/" "$REMOTE_DIR/" "Data directory"
else
    echo -e "${YELLOW}⚠️  No data directory found${NC}"
fi

# Step 5: Upload deployment configurations
echo "5️⃣ Uploading deployment configurations..."
if [ -d "deployment" ]; then
    rsync -av --progress --exclude='.terraform' --exclude='*.tfstate*' \
          "deployment/" "$AWS_HOST:$REMOTE_DIR/deployment/"
    echo "✅ Deployment configurations uploaded"
else
    echo -e "${YELLOW}⚠️  No deployment directory found${NC}"
fi
echo ""

# Step 6: Upload build configuration and requirements
echo "6️⃣ Uploading build files and requirements..."
scp -i "$SSH_KEY" CMakeLists.txt "$AWS_HOST:$REMOTE_DIR/"
if [ -f "requirements.txt" ]; then
    scp -i "$SSH_KEY" requirements.txt "$AWS_HOST:$REMOTE_DIR/"
    echo "✅ requirements.txt uploaded"
else
    echo -e "${YELLOW}⚠️  No requirements.txt found${NC}"
fi
echo ""

# Step 7: Upload tests and benchmarks
upload_with_progress "tests/" "$REMOTE_DIR/" "Tests & Benchmarks (tests/)"

# Step 8: Upload scripts
if [ -d "scripts" ]; then
    upload_with_progress "scripts/" "$REMOTE_DIR/" "Scripts (scripts/)"
fi

# Step 9: Install Python dependencies
echo "9️⃣ Installing Python dependencies..."
ssh -i "$SSH_KEY" "$AWS_HOST" << 'EOF'
cd ~/predis
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python packages..."
    pip3 install --user -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  No requirements.txt found, skipping Python dependency installation"
fi
EOF
echo ""

# Step 10: Verify deployment
echo "🔟 Verifying deployment..."
ssh -i "$SSH_KEY" "$AWS_HOST" << EOF
echo "📁 Deployment structure:"
cd $REMOTE_DIR
find . -maxdepth 2 -type d | sort
echo ""
echo "📊 Disk usage:"
du -sh ./*
EOF
echo ""

# Step 11: Check GPU environment
echo "1️⃣1️⃣ Checking GPU environment..."
ssh -i "$SSH_KEY" "$AWS_HOST" << 'EOF'
echo "🖥️ GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi

echo ""
echo "🔧 CUDA Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep release
else
    echo "⚠️  nvcc not found"
fi
EOF
echo ""

# Step 12: Build Predis
echo "1️⃣2️⃣ Building Predis..."
ssh -i "$SSH_KEY" "$AWS_HOST" << 'EOF'
cd ~/predis
echo "🔨 Cleaning previous builds..."
rm -rf build
mkdir -p build
cd build

echo ""
echo "⚙️ Running CMake configuration..."
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

if cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR \
           -DCMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER \
           -DCMAKE_BUILD_TYPE=Release; then
    echo "✅ CMake configuration successful"
    
    echo ""
    echo "🔧 Building Predis (this may take a few minutes)..."
    if make -j$(nproc); then
        echo "✅ Build successful!"
        echo ""
        echo "📋 Available executables:"
        find . -type f -executable | grep -E "(benchmark|test|predis)" | head -20
    else
        echo "❌ Build failed"
        echo "💡 Checking for specific errors..."
        # Show last few lines of error
        make -j1 2>&1 | tail -50
    fi
else
    echo "❌ CMake configuration failed"
    echo "💡 Checking CMake output..."
    cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR 2>&1 | tail -50
fi
EOF

# Step 13: Setup runtime environment
echo ""
echo "1️⃣3️⃣ Setting up runtime environment..."
ssh -i "$SSH_KEY" "$AWS_HOST" << 'EOF'
cd ~/predis

# Create runtime directories
echo "📁 Creating runtime directories..."
mkdir -p logs models/checkpoints data/cache

# Setup configuration
if [ -f "deployment/config/predis.conf.template" ]; then
    echo "📄 Setting up configuration..."
    cp deployment/config/predis.conf.template predis.conf
    echo "✅ Configuration template copied"
fi

# Create systemd service file (optional)
echo "🔧 Creating systemd service template..."
cat > predis.service << 'SERVICE'
[Unit]
Description=Predis GPU-Accelerated Cache
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/predis
ExecStart=/home/ubuntu/predis/build/predis_server
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

echo "✅ Service file created (not installed)"
echo ""
echo "📝 To install as a service, run:"
echo "   sudo cp predis.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable predis"
echo "   sudo systemctl start predis"
EOF

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Quick Start Guide:"
echo "1. SSH into instance: ssh -i $SSH_KEY $AWS_HOST"
echo "2. Navigate to predis: cd ~/predis"
echo "3. Check build artifacts: ls -la build/"
echo "4. Run tests: cd build && ctest"
echo "5. Run benchmarks: ./build/comprehensive_gpu_benchmark"
echo "6. View Python ML tools: python3 src/ml/examples/feature_engineering_demo.py"
echo ""
echo "🚀 Performance targets:"
echo "   - 10-20x faster than Redis for basic operations"
echo "   - 25-50x faster for batch operations"
echo "   - Billions of ops/sec with GPU acceleration"