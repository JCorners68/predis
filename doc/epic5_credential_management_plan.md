# Epic 5: Secure Cloud Credential Management Implementation Plan

**Timeline**: 2 weeks from start  
**Goal**: Implement enterprise-grade credential management system for Predis  
**Status**: Planning Phase  
**Created**: May 23, 2025

---

## 🚨 Current Security Gap Analysis

After Epic 4 implementation, we have a critical security vulnerability:
- Credentials are still manually managed in deployment scripts
- No automated rotation
- No audit trail for credential access
- No multi-cloud credential synchronization
- Break-glass procedures not defined

This Epic 5 will close these gaps to enable enterprise customer deployments.

---

## 📋 Implementation Roadmap

### Phase 1: AWS Foundation (Week 1, Days 1-3)

#### Day 1: AWS Secrets Manager Integration
```bash
# Create credential storage structure
aws secretsmanager create-secret \
  --name predis/prod/database/password \
  --secret-string "$(openssl rand -base64 32)"

aws secretsmanager create-secret \
  --name predis/prod/grafana/admin \
  --secret-string "$(openssl rand -base64 32)"

aws secretsmanager create-secret \
  --name predis/prod/api/auth-token \
  --secret-string "$(openssl rand -hex 32)"
```

#### Implementation Files to Create:
1. `src/credentials/aws_secrets_manager.cpp`
2. `src/credentials/aws_secrets_manager.h`
3. `scripts/setup_aws_credentials.sh`
4. `scripts/rotate_aws_credentials.sh`

#### Day 2: Terraform State Encryption
```hcl
# terraform/backend.tf
terraform {
  backend "s3" {
    bucket         = "predis-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    kms_key_id     = "alias/predis-terraform"
    dynamodb_table = "predis-terraform-locks"
  }
}

# terraform/secrets.tf
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "predis/${var.environment}/database/password"
}

locals {
  db_password = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)
}
```

#### Day 3: Audit Logging Implementation
```cpp
// src/credentials/audit_logger.cpp
class CredentialAuditLogger {
public:
    void LogAccess(const CredentialAccessEvent& event) {
        // Log to CloudWatch Logs
        CloudWatchEvent cw_event;
        cw_event.timestamp = std::chrono::system_clock::now();
        cw_event.user = event.user_arn;
        cw_event.credential_id = event.credential_id;
        cw_event.action = event.action;  // "read", "rotate", "delete"
        cw_event.source_ip = event.source_ip;
        cw_event.success = event.success;
        
        // Also log to S3 for long-term retention
        S3AuditLog s3_log;
        s3_log.Write(cw_event);
    }
};
```

### Phase 1: Demo Deployment (Days 4-5)

#### Update deploy_customer_secure.sh Integration:
```bash
#!/bin/bash
# deployment/scripts/deploy_customer_secure_v2.sh

# Fetch credentials from AWS Secrets Manager
get_secret() {
    local secret_name=$1
    aws secretsmanager get-secret-value \
        --secret-id "predis/${ENVIRONMENT}/${secret_name}" \
        --query 'SecretString' \
        --output text
}

# No more generated passwords - fetch from Secrets Manager
GRAFANA_ADMIN_PASSWORD=$(get_secret "grafana/admin")
AUTH_TOKEN=$(get_secret "api/auth-token")
DB_PASSWORD=$(get_secret "database/password")

# Deploy with secure credentials
deploy_with_secrets() {
    # Create temporary env file (deleted after deployment)
    cat > "$DEPLOY_DIR/.env.encrypted" <<EOF
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
AUTH_TOKEN=${AUTH_TOKEN}
DB_PASSWORD=${DB_PASSWORD}
EOF
    
    # Encrypt at rest
    openssl enc -aes-256-cbc -salt \
        -in "$DEPLOY_DIR/.env.encrypted" \
        -out "$DEPLOY_DIR/.env.secure" \
        -k "${DEPLOYMENT_KEY}"
    
    # Deploy services
    docker-compose --env-file "$DEPLOY_DIR/.env.secure" up -d
    
    # Clean up
    shred -vfz "$DEPLOY_DIR/.env.encrypted"
}
```

### Phase 2: Multi-Cloud Support (Week 1, Days 6-7 + Week 2, Days 1-2)

#### GCP Secret Manager Integration:
```python
# src/credentials/gcp_secret_manager.py
from google.cloud import secretmanager

class GCPSecretManager:
    def __init__(self, project_id):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
        
    def create_secret(self, secret_id, secret_value):
        parent = f"projects/{self.project_id}"
        
        secret = self.client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )
        
        # Add secret version
        self.client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )
        
    def get_secret(self, secret_id, version="latest"):
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
```

#### Azure Key Vault Integration:
```python
# src/credentials/azure_key_vault.py
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class AzureKeyVaultManager:
    def __init__(self, vault_url):
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
        
    def set_secret(self, name, value):
        self.client.set_secret(name, value)
        
    def get_secret(self, name):
        return self.client.get_secret(name).value
```

#### Cross-Cloud Synchronization:
```cpp
// src/credentials/multi_cloud_sync.cpp
class MultiCloudCredentialSync {
private:
    AWSSecretsManager aws_manager_;
    GCPSecretManager gcp_manager_;
    AzureKeyVault azure_vault_;
    
public:
    void SyncCredential(const std::string& credential_name) {
        // Primary source: AWS
        auto secret = aws_manager_.GetSecret(credential_name);
        
        // Sync to other clouds
        gcp_manager_.SetSecret(credential_name, secret);
        azure_vault_.SetSecret(credential_name, secret);
        
        // Audit log
        AuditLog("credential_sync", credential_name, "aws->gcp,azure");
    }
    
    void RotateAllClouds(const std::string& credential_name) {
        // Generate new credential
        auto new_secret = GenerateSecureCredential();
        
        // Update all clouds atomically
        try {
            aws_manager_.UpdateSecret(credential_name, new_secret);
            gcp_manager_.UpdateSecret(credential_name, new_secret);
            azure_vault_.UpdateSecret(credential_name, new_secret);
            
            AuditLog("credential_rotation", credential_name, "success");
        } catch (const std::exception& e) {
            // Rollback on any failure
            RollbackCredentialUpdate(credential_name);
            throw;
        }
    }
};
```

### Phase 2: Customer Isolation (Week 2, Days 3-4)

#### Customer-Specific Namespace:
```yaml
# kubernetes/customer-secrets.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: predis-customer-${CUSTOMER_ID}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: predis-customer-sa
  namespace: predis-customer-${CUSTOMER_ID}
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/predis-customer-${CUSTOMER_ID}
---
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: customer-secret-store
  namespace: predis-customer-${CUSTOMER_ID}
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: predis-customer-sa
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: customer-credentials
  namespace: predis-customer-${CUSTOMER_ID}
spec:
  secretStoreRef:
    name: customer-secret-store
    kind: SecretStore
  target:
    name: predis-credentials
  data:
  - secretKey: auth-token
    remoteRef:
      key: predis/customer/${CUSTOMER_ID}/auth-token
  - secretKey: grafana-password
    remoteRef:
      key: predis/customer/${CUSTOMER_ID}/grafana-password
```

### Phase 2: Compliance Implementation (Week 2, Days 5-6)

#### SOC2 Compliance Controls:
```python
# src/compliance/soc2_controls.py
class SOC2CredentialControls:
    def __init__(self):
        self.audit_logger = AuditLogger()
        
    def enforce_password_policy(self, password):
        """SOC2 CC6.1 - Logical Access Controls"""
        requirements = {
            'min_length': 16,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True,
            'no_dictionary_words': True,
            'no_repeated_chars': True
        }
        
        if not self.validate_password(password, requirements):
            raise ValueError("Password does not meet SOC2 requirements")
            
    def enforce_rotation_policy(self, credential_type):
        """SOC2 CC6.2 - Periodic Review"""
        rotation_schedule = {
            'user_passwords': 90,  # days
            'api_keys': 30,
            'service_accounts': 180,
            'certificates': 365
        }
        
        return rotation_schedule.get(credential_type, 90)
        
    def audit_credential_access(self, event):
        """SOC2 CC7.2 - System Monitoring"""
        self.audit_logger.log({
            'timestamp': datetime.utcnow(),
            'event_type': 'credential_access',
            'user': event.user,
            'credential_id': event.credential_id,
            'source_ip': event.source_ip,
            'user_agent': event.user_agent,
            'success': event.success,
            'mfa_used': event.mfa_used
        })
```

#### GDPR Compliance:
```cpp
// src/compliance/gdpr_controls.cpp
class GDPRCredentialControls {
public:
    void EncryptPersonalData(Credential& cred) {
        // GDPR Article 32 - Security of processing
        if (cred.contains_personal_data) {
            cred.data = AES256Encrypt(cred.data, GetDataProtectionKey());
            cred.encryption_metadata = {
                {"algorithm", "AES-256-GCM"},
                {"key_id", GetCurrentKeyId()},
                {"encrypted_at", CurrentTimestamp()}
            };
        }
    }
    
    void HandleDeletionRequest(const std::string& customer_id) {
        // GDPR Article 17 - Right to erasure
        auto credentials = GetCustomerCredentials(customer_id);
        
        for (const auto& cred : credentials) {
            // Securely overwrite
            SecureErase(cred);
            
            // Audit log (keep for legal requirements)
            AuditLog("gdpr_deletion", customer_id, cred.id);
        }
    }
};
```

### Phase 3: Production Features (Week 2, Day 7)

#### Automated Rotation Implementation:
```bash
#!/bin/bash
# scripts/automated_rotation.sh

rotate_credential() {
    local credential_type=$1
    local credential_name=$2
    
    case $credential_type in
        "database")
            # Generate new password
            NEW_PASSWORD=$(openssl rand -base64 32)
            
            # Update database first
            mysql -h $DB_HOST -u admin -p$OLD_PASSWORD \
                -e "ALTER USER 'predis'@'%' IDENTIFIED BY '$NEW_PASSWORD';"
            
            # Update secret
            aws secretsmanager update-secret \
                --secret-id "predis/prod/database/password" \
                --secret-string "$NEW_PASSWORD"
            
            # Restart services with new password
            kubectl rollout restart deployment/predis-api
            ;;
            
        "api-key")
            # Generate new API key
            NEW_KEY=$(openssl rand -hex 32)
            
            # Update with version tagging for rollback
            aws secretsmanager update-secret \
                --secret-id "predis/prod/api/key" \
                --secret-string "$NEW_KEY" \
                --version-stage "AWSPENDING"
            
            # Test new key
            if test_api_key "$NEW_KEY"; then
                # Promote to current
                aws secretsmanager update-secret-version-stage \
                    --secret-id "predis/prod/api/key" \
                    --version-stage "AWSCURRENT" \
                    --move-to-version-id "$(get_latest_version)"
            else
                # Rollback
                rollback_secret "predis/prod/api/key"
            fi
            ;;
    esac
}

# Schedule rotations
setup_rotation_schedule() {
    # CloudWatch Events for scheduled rotation
    aws events put-rule \
        --name predis-credential-rotation \
        --schedule-expression "rate(30 days)"
        
    aws events put-targets \
        --rule predis-credential-rotation \
        --targets "Id"="1","Arn"="arn:aws:lambda:region:account:function:rotate-credentials"
}
```

#### Break-Glass Emergency Access:
```python
# src/credentials/break_glass.py
class BreakGlassAccess:
    def __init__(self):
        self.emergency_key_id = "predis-emergency-access"
        
    def request_emergency_access(self, requester_id, reason):
        # Generate time-limited credentials
        emergency_creds = {
            'access_key': generate_temp_key(),
            'expires_at': datetime.utcnow() + timedelta(hours=4),
            'permissions': 'admin',
            'requester': requester_id,
            'reason': reason,
            'approval_required': True
        }
        
        # Send to approval system
        approval_request = self.send_for_approval(emergency_creds)
        
        # Aggressive audit logging
        self.audit_logger.emergency_access_requested({
            'requester': requester_id,
            'reason': reason,
            'request_id': approval_request.id,
            'timestamp': datetime.utcnow()
        })
        
        # Alert security team
        self.alert_security_team(approval_request)
        
        return approval_request
        
    def approve_emergency_access(self, request_id, approver_id):
        # Validate approver has authority
        if not self.is_authorized_approver(approver_id):
            raise UnauthorizedException("Not authorized to approve emergency access")
            
        # Activate credentials
        request = self.get_request(request_id)
        self.activate_emergency_credentials(request.emergency_creds)
        
        # Enhanced audit logging
        self.audit_logger.emergency_access_approved({
            'request_id': request_id,
            'approver': approver_id,
            'activated_at': datetime.utcnow(),
            'expires_at': request.emergency_creds['expires_at']
        })
        
        # Start monitoring session
        self.monitor_emergency_session(request_id)
```

### Phase 3: Monitoring & Alerting (Week 2, Days 7)

#### Security Monitoring Dashboard:
```python
# src/monitoring/credential_security_monitor.py
class CredentialSecurityMonitor:
    def __init__(self):
        self.alert_thresholds = {
            'failed_auth_attempts': 5,
            'after_hours_access': True,
            'unusual_location': True,
            'privilege_escalation': True,
            'rapid_secret_access': 10  # per minute
        }
        
    def monitor_access_patterns(self):
        # Real-time monitoring
        for event in self.get_credential_events():
            # Failed authentication
            if event.type == 'auth_failed':
                self.failed_auth_counter[event.user] += 1
                if self.failed_auth_counter[event.user] > self.alert_thresholds['failed_auth_attempts']:
                    self.alert("Multiple failed auth attempts", event)
                    
            # After hours access
            if event.type == 'credential_access':
                if self.is_after_hours(event.timestamp):
                    self.alert("After hours credential access", event)
                    
            # Unusual location
            if event.type == 'credential_access':
                if not self.is_known_ip(event.source_ip):
                    self.alert("Access from unusual location", event)
                    
            # Rapid access (potential breach)
            if event.type == 'secret_retrieved':
                self.access_rate_limiter.add(event)
                if self.access_rate_limiter.rate > self.alert_thresholds['rapid_secret_access']:
                    self.alert("Rapid secret access detected", event)
                    self.trigger_emergency_response()
```

---

## 🎯 Deliverables Checklist

### Week 1 Deliverables:
- [ ] AWS Secrets Manager integration with rotation
- [ ] Terraform backend encryption and secret injection
- [ ] Audit logging to CloudWatch and S3
- [ ] Updated deployment script with AWS integration
- [ ] Basic monitoring dashboard
- [ ] Demo deployment working with secure credentials

### Week 2 Deliverables:
- [ ] GCP Secret Manager integration
- [ ] Azure Key Vault integration
- [ ] Cross-cloud credential synchronization
- [ ] Customer namespace isolation
- [ ] SOC2 and GDPR compliance controls
- [ ] Automated rotation for all credential types
- [ ] Break-glass emergency procedures
- [ ] Full security monitoring and alerting
- [ ] Complete documentation package

---

## 📚 Documentation Structure

```
doc/security/
├── credential_management/
│   ├── setup_guide.md           # How to set up credentials
│   ├── rotation_procedures.md   # Rotation schedules and processes
│   ├── emergency_access.md      # Break-glass procedures
│   ├── audit_guide.md          # How to audit credential access
│   └── troubleshooting.md      # Common issues and solutions
├── compliance/
│   ├── soc2_controls.md        # SOC2 credential requirements
│   ├── gdpr_compliance.md      # GDPR data protection
│   ├── pci_dss_requirements.md # PCI-DSS for payments
│   └── hipaa_safeguards.md    # HIPAA for healthcare
└── architecture/
    ├── credential_flow.md      # How credentials flow through system
    ├── security_model.md       # Zero-trust architecture
    └── threat_model.md        # Security threat analysis
```

---

## 🚀 Implementation Priority

1. **Immediate (Days 1-3)**: AWS integration for trial deployments
2. **Critical (Days 4-5)**: Demo environment with full security
3. **Important (Days 6-9)**: Multi-cloud and compliance
4. **Enhancement (Days 10-14)**: Advanced monitoring and automation

This implementation will transform Predis from having basic security to enterprise-grade credential management suitable for Fortune 500 deployments.