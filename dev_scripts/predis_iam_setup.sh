#!/bin/zsh
# setup_predis_iam_complete.sh
# Complete IAM setup script for Predis GPU deployment with granular permissions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
USER_NAME="predis_hw"
POLICY_PREFIX="Predis"

echo -e "${BLUE}🚀 Setting up comprehensive IAM policies for Predis deployment${NC}"
echo "User: $USER_NAME"
echo "Account: $(aws sts get-caller-identity --query Account --output text)"
echo

# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Function to create policy file
create_policy_file() {
    local policy_name=$1
    local policy_content=$2
    local filename="${policy_name,,}-policy.json"
    
    echo "$policy_content" > "$filename"
    echo -e "${GREEN}✓${NC} Created policy file: $filename"
}

# Function to create and attach policy
create_and_attach_policy() {
    local policy_name=$1
    local filename=$2
    local description=$3
    
    echo -e "${YELLOW}Creating policy: $policy_name${NC}"
    
    # Create policy (ignore error if exists)
    aws iam create-policy \
        --policy-name "$policy_name" \
        --policy-document "file://$filename" \
        --description "$description" \
        --output text > /dev/null 2>&1 || echo "  Policy may already exist"
    
    # Attach to user
    aws iam attach-user-policy \
        --user-name "$USER_NAME" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$policy_name" || echo "  Policy may already be attached"
    
    echo -e "${GREEN}✓${NC} Policy $policy_name attached to $USER_NAME"
}

echo -e "${BLUE}📋 Creating IAM policy files...${NC}"

# 1. EC2 GPU Management Policy
create_policy_file "PredisEC2GPU" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EC2InstanceManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:StopInstances",
                "ec2:StartInstances",
                "ec2:RebootInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeInstanceTypeOfferings",
                "ec2:DescribeImages",
                "ec2:DescribeSnapshots",
                "ec2:DescribeVolumes",
                "ec2:DescribeAccountAttributes",
                "ec2:DescribeAvailabilityZones",
                "ec2:DescribeRegions",
                "ec2:DescribeSubnets",
                "ec2:DescribeVpcs",
                "ec2:DescribeInstanceAttribute",
                "ec2:ModifyInstanceAttribute"
            ],
            "Resource": "*"
        },
        {
            "Sid": "EC2NetworkingManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:CreateSecurityGroup",
                "ec2:DeleteSecurityGroup",
                "ec2:DescribeSecurityGroups",
                "ec2:AuthorizeSecurityGroupIngress",
                "ec2:AuthorizeSecurityGroupEgress",
                "ec2:RevokeSecurityGroupIngress",
                "ec2:RevokeSecurityGroupEgress",
                "ec2:CreateTags",
                "ec2:DescribeTags",
                "ec2:DeleteTags"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "ec2:ResourceTag/Project": "predis*"
                }
            }
        },
        {
            "Sid": "EC2KeyPairManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:CreateKeyPair",
                "ec2:DeleteKeyPair",
                "ec2:DescribeKeyPairs",
                "ec2:ImportKeyPair"
            ],
            "Resource": "*"
        },
        {
            "Sid": "EC2VolumeManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:CreateVolume",
                "ec2:DeleteVolume",
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:ModifyVolume",
                "ec2:DescribeVolumeStatus",
                "ec2:DescribeVolumeAttribute"
            ],
            "Resource": "*"
        },
        {
            "Sid": "EC2SpotInstances",
            "Effect": "Allow",
            "Action": [
                "ec2:RequestSpotInstances",
                "ec2:CancelSpotInstanceRequests",
                "ec2:DescribeSpotInstanceRequests",
                "ec2:DescribeSpotPriceHistory",
                "ec2:DescribeSpotFleetInstances",
                "ec2:DescribeSpotFleetRequests",
                "ec2:RequestSpotFleet",
                "ec2:CancelSpotFleetRequests"
            ],
            "Resource": "*"
        }
    ]
}'

# 2. Service Quotas and Billing Policy
create_policy_file "PredisServiceQuotas" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ServiceQuotasManagement",
            "Effect": "Allow",
            "Action": [
                "servicequotas:GetServiceQuota",
                "servicequotas:ListServiceQuotas",
                "servicequotas:GetRequestedServiceQuotaChange",
                "servicequotas:RequestServiceQuotaIncrease",
                "servicequotas:ListRequestedServiceQuotaChangeHistory",
                "servicequotas:ListServices"
            ],
            "Resource": "*"
        },
        {
            "Sid": "BillingAndCostAccess",
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetUsageReport",
                "ce:GetDimensionValues",
                "ce:GetMetrics",
                "budgets:CreateBudget",
                "budgets:UpdateBudget",
                "budgets:ViewBudget",
                "budgets:DescribeBudgets"
            ],
            "Resource": "*"
        }
    ]
}'

# 3. IAM Role Management (Limited)
create_policy_file "PredisIAMRoles" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "IAMRoleManagement",
            "Effect": "Allow",
            "Action": [
                "iam:PassRole",
                "iam:GetRole",
                "iam:CreateRole",
                "iam:DeleteRole",
                "iam:AttachRolePolicy",
                "iam:DetachRolePolicy",
                "iam:CreateInstanceProfile",
                "iam:DeleteInstanceProfile",
                "iam:AddRoleToInstanceProfile",
                "iam:RemoveRoleFromInstanceProfile",
                "iam:ListInstanceProfiles",
                "iam:GetInstanceProfile",
                "iam:TagRole",
                "iam:UntagRole"
            ],
            "Resource": [
                "arn:aws:iam::*:role/predis-*",
                "arn:aws:iam::*:instance-profile/predis-*"
            ]
        },
        {
            "Sid": "IAMPolicyManagement",
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:DeletePolicy",
                "iam:GetPolicy",
                "iam:ListPolicies",
                "iam:CreatePolicyVersion",
                "iam:DeletePolicyVersion"
            ],
            "Resource": [
                "arn:aws:iam::*:policy/predis-*",
                "arn:aws:iam::*:policy/Predis*"
            ]
        }
    ]
}'

# 4. CloudFormation and AutoScaling
create_policy_file "PredisInfrastructure" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CloudFormationManagement",
            "Effect": "Allow",
            "Action": [
                "cloudformation:CreateStack",
                "cloudformation:UpdateStack",
                "cloudformation:DeleteStack",
                "cloudformation:DescribeStacks",
                "cloudformation:DescribeStackEvents",
                "cloudformation:DescribeStackResources",
                "cloudformation:DescribeStackResource",
                "cloudformation:ListStacks",
                "cloudformation:GetTemplate",
                "cloudformation:ValidateTemplate",
                "cloudformation:EstimateTemplateCost",
                "cloudformation:CreateChangeSet",
                "cloudformation:DescribeChangeSet",
                "cloudformation:ExecuteChangeSet"
            ],
            "Resource": "*",
            "Condition": {
                "StringLike": {
                    "cloudformation:StackName": "predis-*"
                }
            }
        },
        {
            "Sid": "AutoScalingManagement",
            "Effect": "Allow",
            "Action": [
                "autoscaling:CreateAutoScalingGroup",
                "autoscaling:DeleteAutoScalingGroup",
                "autoscaling:DescribeAutoScalingGroups",
                "autoscaling:UpdateAutoScalingGroup",
                "autoscaling:CreateLaunchTemplate",
                "autoscaling:DeleteLaunchTemplate",
                "autoscaling:DescribeLaunchTemplates",
                "autoscaling:CreateOrUpdateTags",
                "autoscaling:DeleteTags"
            ],
            "Resource": "*"
        },
        {
            "Sid": "LaunchTemplateManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:CreateLaunchTemplate",
                "ec2:DeleteLaunchTemplate",
                "ec2:DescribeLaunchTemplates",
                "ec2:DescribeLaunchTemplateVersions",
                "ec2:ModifyLaunchTemplate",
                "ec2:CreateLaunchTemplateVersion"
            ],
            "Resource": "*"
        }
    ]
}'

# 5. Monitoring and Logging
create_policy_file "PredisMonitoring" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CloudWatchMonitoring",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:GetMetricData",
                "cloudwatch:ListMetrics",
                "cloudwatch:DescribeAlarms",
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:DeleteAlarms",
                "cloudwatch:CreateDashboard",
                "cloudwatch:DeleteDashboard",
                "cloudwatch:GetDashboard",
                "cloudwatch:ListDashboards"
            ],
            "Resource": "*"
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams",
                "logs:GetLogEvents",
                "logs:FilterLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:*:*:log-group:/predis/*",
                "arn:aws:logs:*:*:log-group:predis-*"
            ]
        }
    ]
}'

# 6. Secrets and Configuration Management
create_policy_file "PredisSecrets" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SecretsManagerAccess",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:CreateSecret",
                "secretsmanager:GetSecretValue",
                "secretsmanager:PutSecretValue",
                "secretsmanager:UpdateSecret",
                "secretsmanager:DeleteSecret",
                "secretsmanager:DescribeSecret",
                "secretsmanager:ListSecrets",
                "secretsmanager:RotateSecret",
                "secretsmanager:TagResource",
                "secretsmanager:UntagResource"
            ],
            "Resource": [
                "arn:aws:secretsmanager:*:*:secret:predis/*",
                "arn:aws:secretsmanager:*:*:secret:predis-*"
            ]
        },
        {
            "Sid": "ParameterStoreAccess",
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter",
                "ssm:GetParameters",
                "ssm:PutParameter",
                "ssm:DeleteParameter",
                "ssm:DescribeParameters",
                "ssm:GetParameterHistory",
                "ssm:AddTagsToResource",
                "ssm:RemoveTagsFromResource"
            ],
            "Resource": [
                "arn:aws:ssm:*:*:parameter/predis/*",
                "arn:aws:ssm:*:*:parameter/predis-*"
            ]
        }
    ]
}'

# 7. S3 Storage Policy
create_policy_file "PredisS3" '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3DeploymentBuckets",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:DeleteBucket",
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetBucketVersioning",
                "s3:PutBucketVersioning",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:GetObjectVersion",
                "s3:PutObjectAcl",
                "s3:GetObjectAcl",
                "s3:PutBucketTagging",
                "s3:GetBucketTagging"
            ],
            "Resource": [
                "arn:aws:s3:::predis-*",
                "arn:aws:s3:::predis-*/*"
            ]
        }
    ]
}'

echo -e "${BLUE}🔧 Creating and attaching policies...${NC}"

# Create and attach all policies
create_and_attach_policy "PredisEC2GPU" "predisec2gpu-policy.json" "EC2 and GPU instance management for Predis"
create_and_attach_policy "PredisServiceQuotas" "predisservicequotas-policy.json" "Service quotas and billing access for Predis"
create_and_attach_policy "PredisIAMRoles" "predisiamroles-policy.json" "Limited IAM role management for Predis"
create_and_attach_policy "PredisInfrastructure" "predisinfrastructure-policy.json" "CloudFormation and infrastructure management for Predis"
create_and_attach_policy "PredisMonitoring" "predismonitoring-policy.json" "CloudWatch monitoring and logging for Predis"
create_and_attach_policy "PredisSecrets" "predissecrets-policy.json" "Secrets and configuration management for Predis"
create_and_attach_policy "PredisS3" "prediss3-policy.json" "S3 storage for Predis deployment artifacts"

echo
echo -e "${BLUE}📋 Current attached policies for $USER_NAME:${NC}"
aws iam list-attached-user-policies --user-name "$USER_NAME" --output table

echo
echo -e "${YELLOW}🧪 Testing permissions...${NC}"

# Test key permissions
echo "Testing EC2 permissions..."
aws ec2 describe-regions --output text > /dev/null && echo -e "${GREEN}✓${NC} EC2 access working"

echo "Testing Service Quotas permissions..."
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1 > /dev/null && echo -e "${GREEN}✓${NC} Service Quotas access working"

echo "Testing spot pricing..."
aws ec2 describe-spot-price-history --instance-types g4dn.xlarge --max-items 1 --region us-east-1 > /dev/null && echo -e "${GREEN}✓${NC} Spot pricing access working"

echo
echo -e "${GREEN}🎉 IAM setup complete!${NC}"
echo
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Test GPU instance launch with: aws ec2 run-instances --image-id <ami> --instance-type g4dn.xlarge --count 1 --dry-run"
echo "2. Remove temporary broad permissions:"
echo "   aws iam detach-user-policy --user-name $USER_NAME --policy-arn arn:aws:iam::aws:policy/PowerUserAccess"
echo "   aws iam detach-user-policy --user-name $USER_NAME --policy-arn arn:aws:iam::aws:policy/IAMFullAccess"
echo "3. Keep AmazonEC2FullAccess for now (can replace with PredisEC2GPU later)"
echo
echo -e "${BLUE}💰 Cost Monitoring:${NC}"
echo "Set up a budget alert:"
echo "aws budgets create-budget --account-id $ACCOUNT_ID --budget '{\"BudgetName\":\"Predis-Monthly\",\"BudgetLimit\":{\"Amount\":\"200\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}'"
echo
echo -e "${BLUE}🔒 Security:${NC}"
echo "All policies are scoped to predis-* resources where possible"
echo "IAM permissions limited to predis-* roles and policies"
echo "S3 access limited to predis-* buckets"
echo
echo -e "${RED}⚠️  Remember to clean up instances when not in use to control costs!${NC}"