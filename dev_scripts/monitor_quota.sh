#!/bin/bash
echo "=== G4DN Quota Status Check ==="
echo "Date: $(date)"

QUOTA=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1 --query 'Quota.Value' --output text)
echo "Current Quota: $QUOTA vCPUs"

if [[ "$QUOTA" != "0.0" ]]; then
    echo "🎉 QUOTA APPROVED! Ready to launch GPU instances!"
    echo "Run: ./dev_scripts/quick_test.sh"
else
    echo "⏳ Still waiting for approval..."
    aws service-quotas get-requested-service-quota-change --request-id c9c103fef14a403091b1ce4b15851ba5ZImbyLmp --region us-east-1 --query 'RequestedQuota.Status'
fi
