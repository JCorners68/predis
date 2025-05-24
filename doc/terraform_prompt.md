# Epic 4.1 - Terraform AWS Deployment Script
## Simple Infrastructure as Code for Predis MLOps Platform

### Overview
Create a **single, simple Terraform script** that deploys Predis to AWS with minimal complexity but production-grade security. Focus on getting it working quickly rather than perfect modularity.

### Requirements

#### Core Infrastructure Components
```hcl
# Pseudo-code structure:
resource "aws_vpc" "predis_vpc" {
  # CIDR: 10.0.0.0/16
  # Enable DNS hostnames and resolution
}

resource "aws_subnet" "predis_public" {
  # CIDR: 10.0.1.0/24
  # Map public IP on launch
  # Availability zone: us-east-1a
}

resource "aws_internet_gateway" "predis_igw" {
  # Attach to VPC
}

resource "aws_route_table" "predis_public_rt" {
  vpc_id = aws_vpc.predis_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.predis_igw.id
  }

  tags = {
    Name = "predis-public-rt"
  }
}

resource "aws_route_table_association" "predis_public_rta" {
  subnet_id      = aws_subnet.predis_public.id
  route_table_id = aws_route_table.predis_public_rt.id
}

resource "aws_security_group" "predis_sg" {
  # Inbound: 22 (SSH), 8080 (API), 3000 (Dashboard), 9090 (Metrics)
  # Outbound: All traffic
  # Source: Your IP only (not 0.0.0.0/0)
}

resource "aws_key_pair" "predis_key" {
  # Use existing public key or generate
}

resource "aws_instance" "predis_gpu" {
  # Instance: g4dn.xlarge
  # AMI: Deep Learning AMI GPU PyTorch (Ubuntu 22.04)
  # User data: Docker + CUDA setup script
}
```

#### Security & IAM Components
```hcl
resource "aws_iam_role" "predis_instance_role" {
  # EC2 service principal
  # Minimal permissions for CloudWatch, S3 access
}

resource "aws_iam_instance_profile" "predis_profile" {
  # Attach role to EC2 instance
}

resource "aws_iam_policy" "predis_policy" {
  # CloudWatch logs write
  # S3 bucket read/write (for model artifacts)
  # EC2 describe (for auto-scaling metadata)
}
```

#### Storage Components
```hcl
resource "aws_s3_bucket" "predis_artifacts" {
  bucket = "predis-artifacts-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket_versioning" "predis_artifacts_versioning" {
  bucket = aws_s3_bucket.predis_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "predis_artifacts_encryption" {
  bucket = aws_s3_bucket.predis_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}
resource "aws_ebs_volume" "predis_data" {
  availability_zone = aws_instance.predis_gpu.availability_zone
  size              = 100
  type              = "gp3"
  encrypted         = true
  
  tags = {
    Name = "predis-data"
  }
}

resource "aws_volume_attachment" "predis_data_attachment" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.predis_data.id
  instance_id = aws_instance.predis_gpu.id
}
```

### Critical Implementation Details

#### 1. AMI Selection
```hcl
# Use AWS Deep Learning AMI - has CUDA pre-installed
data "aws_ami" "deep_learning_ami" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*Ubuntu 22.04*"]
  }
  
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}
```

#### 2. User Data Script (Bootstrap)
```bash
#!/bin/bash
# Install Docker with GPU support
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://nvidia.github.io/nvidia-docker/ubuntu22.04/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Clone and deploy Predis
cd /home/ubuntu
git clone https://github.com/your-username/predis.git
cd predis/deployment/docker
docker build -t predis:epic4 .
docker run -d --name predis --gpus all -p 8080:8080 -p 3000:3000 -p 9090:9090 predis:epic4

# Setup logging to CloudWatch
apt-get install -y awslogs
# Configure awslogs.conf for application logs
```

#### 3. Security Group Rules (Restrictive)
```hcl
# Get your current IP automatically
data "http" "myip" {
  url = "http://ipv4.icanhazip.com"
}

locals {
  my_ip = "${chomp(data.http.myip.body)}/32"
}

# Security group with restricted access
resource "aws_security_group" "predis_sg" {
  name_prefix = "predis-sg-"
  vpc_id      = aws_vpc.predis_vpc.id

  # SSH - Your IP only
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.my_ip]
  }

  # API - Your IP only (for now)
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [local.my_ip]
  }

  # Dashboard - Your IP only
  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [local.my_ip]
  }

  # Metrics - Your IP only
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [local.my_ip]
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

#### 4. Variables & Outputs
```hcl
variable "region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "EC2 Key Pair name"
  type        = string
}

variable "project_name" {
  description = "Project name for tagging"
  default     = "predis"
}

output "instance_ip" {
  value = aws_instance.predis_gpu.public_ip
}

output "dashboard_url" {
  value = "http://${aws_instance.predis_gpu.public_ip}:3000"
}

output "api_url" {
  value = "http://${aws_instance.predis_gpu.public_ip}:8080"
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.predis_gpu.public_ip}"
}
```

### File Structure
```
deployment/terraform/
├── main.tf           # Single file with all resources
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── terraform.tfvars.example  # Example variables
└── README.md         # Deployment instructions
```

### Deployment Commands
```bash
# Initialize
terraform init

# Plan (review changes)
terraform plan -var="key_name=your-key-name"

# Apply
terraform apply -var="key_name=your-key-name" -auto-approve

# Get outputs
terraform output

# Destroy when done
terraform destroy -auto-approve
```

### Required AWS Permissions
The deploying user needs these IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "iam:CreateRole",
        "iam:CreateInstanceProfile",
        "iam:CreatePolicy",
        "iam:AttachRolePolicy",
        "iam:PassRole",
        "s3:CreateBucket",
        "s3:PutBucketVersioning",
        "s3:PutBucketEncryption"
      ],
      "Resource": "*"
    }
  ]
}
```

### Cost Optimization
```hcl
# Instance scheduling (optional)
resource "aws_instance" "predis_gpu" {
  # ... other config
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    # Auto-shutdown after 4 hours to save costs
    shutdown_timer = "240"  # minutes
  }))
  
  # Use Spot instances for 70% cost savings (optional)
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.40"  # vs $0.526 on-demand
    }
  }
}
```

### Validation Steps
```bash
# After deployment, validate:
# 1. SSH access works
ssh -i ~/.ssh/your-key.pem ubuntu@<INSTANCE_IP>

# 2. GPU is accessible
nvidia-smi

# 3. Docker is running
docker ps

# 4. Predis services respond
curl http://<INSTANCE_IP>:8080/health
curl http://<INSTANCE_IP>:3000
```

### Epic 4.1 Success Criteria
- ✅ Single command deployment (`terraform apply`)
- ✅ GPU instance with CUDA working
- ✅ Docker containers running Predis
- ✅ All ports accessible from your IP
- ✅ Proper IAM roles and security groups
- ✅ S3 bucket for model artifacts
- ✅ CloudWatch logging configured
- ✅ Clean teardown (`terraform destroy`)

### Time Estimate: 20 minutes
- 5 min: Write terraform script
- 10 min: Test deployment
- 5 min: Validate services

This gives you **true Infrastructure as Code** for your Series A demo while keeping complexity minimal for the 30-minute deadline.