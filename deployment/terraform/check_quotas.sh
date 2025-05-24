#!/bin/bash
# check_quotas.sh - Check AWS quotas and deployment readiness for Predis

set -e

echo "=== AWS Quota and Deployment Readiness Check ==="
echo "Checking quotas for Predis GPU development environment..."
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ Error: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

# Get current account and region info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${1:-$(aws configure get region || echo "us-east-1")}

echo "🔍 Checking account: $ACCOUNT_ID in region: $REGION"
echo ""

# Function to check service quota
check_quota() {
    local service_code=$1
    local quota_code=$2
    local quota_name=$3
    local needed_value=$4
    
    echo "📊 Checking: $quota_name"
    
    # Get current quota value
    quota_value=$(aws service-quotas get-service-quota \
        --service-code "$service_code" \
        --quota-code "$quota_code" \
        --region "$REGION" \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "ERROR")
    
    if [ "$quota_value" = "ERROR" ]; then
        echo "   ⚠️  Could not retrieve quota information"
        return 1
    fi
    
    # Convert to integer for comparison
    quota_int=$(printf "%.0f" "$quota_value")
    
    if [ "$quota_int" -ge "$needed_value" ]; then
        echo "   ✅ Current limit: $quota_int (needed: $needed_value)"
        return 0
    else
        echo "   ❌ Current limit: $quota_int (needed: $needed_value) - INSUFFICIENT"
        return 1
    fi
}

# Function to get current usage
get_current_usage() {
    local instance_type=$1
    echo "📈 Current usage for $instance_type:"
    
    # Count running instances of this type
    running_count=$(aws ec2 describe-instances \
        --filters "Name=instance-type,Values=$instance_type" \
                  "Name=instance-state-name,Values=running,pending" \
        --query 'length(Reservations[].Instances[])' \
        --region "$REGION" \
        --output text 2>/dev/null || echo "0")
    
    echo "   🖥️  Running/Pending instances: $running_count"
    
    # Show all GPU instances if any
    gpu_instances=$(aws ec2 describe-instances \
        --filters "Name=instance-type,Values=g4dn.*,g5.*,p3.*,p4d.*" \
                  "Name=instance-state-name,Values=running,pending,stopping" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' \
        --region "$REGION" \
        --output table 2>/dev/null || echo "")
    
    if [ -n "$gpu_instances" ] && [ "$gpu_instances" != "" ]; then
        echo "   🎮 All GPU instances in account:"
        echo "$gpu_instances"
    else
        echo "   🎮 No GPU instances currently running"
    fi
}

# Check critical EC2 quotas
echo "🚀 Checking EC2 Quotas"
echo "===================="

# Check G and VT instance quota (this is what you got approved)
all_passed=true

if ! check_quota "ec2" "L-DB2E81BA" "All G and VT Spot Instance Requests" 4; then
    all_passed=false
fi

# Alternative quota check - Running On-Demand G instances
if ! check_quota "ec2" "L-3819A6DF" "Running On-Demand G instances" 4; then
    all_passed=false
fi

echo ""

# Check vCPU quotas
echo "💻 Checking vCPU Quotas"
echo "======================"

# Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances
if ! check_quota "ec2" "L-1216C47A" "Running On-Demand Standard instances" 32; then
    all_passed=false
fi

echo ""

# Check EBS quotas
echo "💾 Checking EBS Quotas"
echo "====================="

# General Purpose SSD (gp3) volume storage
if ! check_quota "ebs" "L-7A658B76" "General Purpose SSD (gp3) volume storage" 1000; then
    all_passed=false
fi

echo ""

# Check VPC quotas
echo "🌐 Checking VPC Quotas"
echo "====================="

# Security groups per VPC
if ! check_quota "vpc" "L-E79EC296" "Security groups per VPC" 10; then
    all_passed=false
fi

# VPCs per Region
if ! check_quota "vpc" "L-F678F1CE" "VPCs per Region" 5; then
    all_passed=false
fi

echo ""

# Check Elastic IP quotas
echo "🔗 Checking Elastic IP Quotas"
echo "============================="

# EC2-VPC Elastic IPs
if ! check_quota "ec2" "L-0263D0A3" "EC2-VPC Elastic IPs" 5; then
    all_passed=false
fi

echo ""

# Check current usage
echo "📊 Current Usage Analysis"
echo "========================"
get_current_usage "g4dn.xlarge"

echo ""

# Check specific instance type availability in current region
echo "🌍 Instance Type Availability"
echo "============================="

GPU_INSTANCE_TYPES=("g4dn.xlarge" "g4dn.2xlarge" "g5.xlarge" "g5.2xlarge" "p3.2xlarge")

for instance_type in "${GPU_INSTANCE_TYPES[@]}"; do
    echo "🔍 Checking availability: $instance_type"
    
    zones=$(aws ec2 describe-instance-type-offerings \
        --location-type availability-zone \
        --filters "Name=instance-type,Values=$instance_type" \
        --query 'InstanceTypeOfferings[].Location' \
        --region "$REGION" \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$zones" ]; then
        echo "   ✅ Available in zones: $zones"
    else
        echo "   ❌ Not available in region $REGION"
    fi
done

echo ""

# Check pricing for cost estimation
echo "💰 Cost Estimation (approximate)"
echo "================================"

echo "Estimated hourly costs in $REGION:"
echo "   g4dn.xlarge:  ~$0.526/hour  (~$12.60/day)"
echo "   g4dn.2xlarge: ~$0.752/hour  (~$18.05/day)"
echo "   g5.xlarge:    ~$1.006/hour  (~$24.14/day)"
echo "   g5.2xlarge:   ~$1.212/hour  (~$29.09/day)"
echo ""
echo "💡 Tip: Use AWS Pricing Calculator for exact pricing"

echo ""

# Final summary
echo "📋 Deployment Readiness Summary"
echo "==============================="

if [ "$all_passed" = true ]; then
    echo "✅ ALL QUOTAS SUFFICIENT - Ready to deploy!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. cd ~/predis/deployment/terraform"
    echo "   2. terraform init"
    echo "   3. terraform plan"
    echo "   4. terraform apply"
    echo ""
    echo "⏱️  Expected deployment time: 10-15 minutes"
    echo "💰 Estimated cost: ~$12.60/day for g4dn.xlarge"
else
    echo "❌ SOME QUOTAS INSUFFICIENT - Request increases before deploying"
    echo ""
    echo "🔧 To request quota increases:"
    echo "   1. Visit: https://console.aws.amazon.com/servicequotas/home"
    echo "   2. Select the service (EC2, EBS, VPC)"
    echo "   3. Find the quota and click 'Request quota increase'"
    echo "   4. Justify the increase (mention 'GPU development for ML project')"
fi

echo ""
echo "📖 Additional Resources:"
echo "   - Service Quotas Console: https://console.aws.amazon.com/servicequotas/home"
echo "   - EC2 Instance Types: https://aws.amazon.com/ec2/instance-types/"
echo "   - AWS Pricing Calculator: https://calculator.aws/"

# Save quota check results
cat > quota_check_results.json << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "account_id": "$ACCOUNT_ID",
  "region": "$REGION",
  "deployment_ready": $all_passed,
  "checked_quotas": [
    "All G and VT Spot Instance Requests",
    "Running On-Demand G instances", 
    "Running On-Demand Standard instances",
    "General Purpose SSD (gp3) volume storage",
    "Security groups per VPC",
    "VPCs per Region",
    "EC2-VPC Elastic IPs"
  ]
}
EOF

echo ""
echo "💾 Results saved to quota_check_results.json"