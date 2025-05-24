# Create SSH key if you haven't already
aws ec2 create-key-pair --key-name predis-demo-key --query 'KeyMaterial' --output text > predis-demo-key.pem
chmod 400 predis-demo-key.pem

# Add SSH access to the default security group
YOUR_IP=$(curl -s https://ipinfo.io/ip)
echo "Your IP: $YOUR_IP"

aws ec2 authorize-security-group-ingress \
  --group-id sg-99f3b697 \
  --protocol tcp \
  --port 22 \
  --cidr $YOUR_IP/32 \
  --region us-east-1

echo "✓ SSH access added to security group"

# Launch G4DN instance
echo "🚀 Launching G4DN.xlarge GPU instance..."

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-09f4814ae750baed6 \
  --instance-type g4dn.xlarge \
  --key-name predis-demo-key \
  --security-group-ids sg-99f3b697 \
  --count 1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=predis-gpu-demo},{Key=Project,Value=predis}]' \
  --region us-east-1 \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "✓ Instance launched: $INSTANCE_ID"
echo "💰 Cost: ~$0.53/hour"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region us-east-1)

echo "🎉 GPU Instance is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Instance Type: g4dn.xlarge (Tesla T4 GPU, 16GB VRAM)"
echo "SSH Command: ssh -i predis-demo-key.pem ec2-user@$PUBLIC_IP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Next: Wait 2-3 minutes for boot, then SSH in and run 'nvidia-smi'"