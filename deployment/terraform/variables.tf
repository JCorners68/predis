variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "disk_size_gb" {
  description = "Root disk size in GB"
  type        = number
  default     = 30
}

variable "public_key_path" {
  description = "Path to public SSH key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "private_key_path" {
  description = "Path to private SSH key"
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "use_spot_instances" {
  description = "Use spot instances"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for spot instances"
  type        = string
  default     = "0.50"
}