terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_security_group" "predis_dev" {
  name_prefix = "predis-dev-"
  description = "Security group for Predis development"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8080
    to_port     = 8090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "predis-dev-sg"
  }
}

resource "aws_key_pair" "predis_key" {
  key_name   = "predis-dev-key-${random_id.suffix.hex}"
  public_key = file(var.public_key_path)
}

resource "aws_instance" "predis_dev" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.predis_key.key_name
  vpc_security_group_ids = [aws_security_group.predis_dev.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  dynamic "instance_market_options" {
    for_each = var.use_spot_instances ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        instance_interruption_behavior = "terminate"
        max_price                      = var.spot_max_price
        spot_instance_type             = "one-time"
      }
    }
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.disk_size_gb
    encrypted   = true
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y nvidia-driver-535 docker.io
    usermod -aG docker ubuntu
    echo "Setup complete" > /home/ubuntu/setup_complete.txt
    EOF
  )

  tags = {
    Name = "predis-dev-instance"
  }
}

resource "aws_eip" "predis_dev" {
  instance = aws_instance.predis_dev.id
  domain   = "vpc"
}

output "instance_id" {
  value = aws_instance.predis_dev.id
}

output "public_ip" {
  value = aws_eip.predis_dev.public_ip
}

output "ssh_command" {
  value = "ssh -i ${var.private_key_path} ubuntu@${aws_eip.predis_dev.public_ip}"
}