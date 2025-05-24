#!/bin/bash

# Predis Customer Deployment Script - Secure Version
# Automates deployment for pilot customers with monitoring and validation

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CUSTOMER_ID=""
DEPLOYMENT_TYPE="docker"  # docker, kubernetes, aws
INSTANCE_TYPE="g4dn.xlarge"
CACHE_MEMORY_GB=8
MLOPS_ENABLED="true"
MONITORING_INTEGRATION="prometheus"
DRY_RUN=false
GRAFANA_ADMIN_PASSWORD=""
AUTH_TOKEN=""

# Security functions
sanitize_input() {
    # Remove potentially dangerous characters
    local input="$1"
    # Allow only alphanumeric, dash, underscore
    echo "$input" | sed 's/[^a-zA-Z0-9_-]//g'
}

validate_customer_id() {
    local id="$1"
    if [[ ! "$id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{2,49}$ ]]; then
        log_error "Invalid customer ID. Must be 3-50 characters, alphanumeric with dash/underscore only"
        exit 1
    fi
}

generate_secure_password() {
    # Generate cryptographically secure password
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
}

generate_auth_token() {
    # Generate secure auth token
    openssl rand -hex 32
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --customer-id)
            CUSTOMER_ID="$(sanitize_input "$2")"
            shift 2
            ;;
        --deployment-type)
            DEPLOYMENT_TYPE="$2"
            if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|kubernetes|aws)$ ]]; then
                log_error "Invalid deployment type. Must be: docker, kubernetes, or aws"
                exit 1
            fi
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --cache-memory)
            CACHE_MEMORY_GB="$2"
            if [[ ! "$CACHE_MEMORY_GB" =~ ^[0-9]+$ ]] || [ "$CACHE_MEMORY_GB" -lt 1 ] || [ "$CACHE_MEMORY_GB" -gt 64 ]; then
                log_error "Cache memory must be between 1 and 64 GB"
                exit 1
            fi
            shift 2
            ;;
        --no-mlops)
            MLOPS_ENABLED="false"
            shift
            ;;
        --monitoring)
            MONITORING_INTEGRATION="$2"
            if [[ ! "$MONITORING_INTEGRATION" =~ ^(prometheus|datadog|cloudwatch|newrelic)$ ]]; then
                log_error "Invalid monitoring integration"
                exit 1
            fi
            shift 2
            ;;
        --grafana-password)
            GRAFANA_ADMIN_PASSWORD="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 --customer-id <id> [options]"
            echo "Options:"
            echo "  --deployment-type   docker|kubernetes|aws (default: docker)"
            echo "  --instance-type     GPU instance type (default: g4dn.xlarge)"
            echo "  --cache-memory      GPU memory for cache in GB (default: 8)"
            echo "  --no-mlops          Disable MLOps pipeline"
            echo "  --monitoring        prometheus|datadog|cloudwatch|newrelic (default: prometheus)"
            echo "  --grafana-password  Admin password for Grafana (auto-generated if not provided)"
            echo "  --dry-run           Show what would be deployed without deploying"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$CUSTOMER_ID" ]; then
    echo -e "${RED}Error: --customer-id is required${NC}"
    exit 1
fi

validate_customer_id "$CUSTOMER_ID"

# Generate secure passwords if not provided
if [ -z "$GRAFANA_ADMIN_PASSWORD" ]; then
    GRAFANA_ADMIN_PASSWORD=$(generate_secure_password)
    log_info "Generated Grafana admin password (save this): $GRAFANA_ADMIN_PASSWORD"
fi

AUTH_TOKEN=$(generate_auth_token)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for required tools
    for tool in openssl sed; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    case $DEPLOYMENT_TYPE in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker not found. Please install Docker."
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose not found. Please install Docker Compose."
                exit 1
            fi
            # Check for NVIDIA Docker runtime
            if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
                log_error "NVIDIA Docker runtime not available. Please install nvidia-docker2."
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl not found. Please install kubectl."
                exit 1
            fi
            if ! command -v helm &> /dev/null; then
                log_error "Helm not found. Please install Helm."
                exit 1
            fi
            ;;
        aws)
            if ! command -v aws &> /dev/null; then
                log_error "AWS CLI not found. Please install AWS CLI."
                exit 1
            fi
            # Check AWS credentials
            if ! aws sts get-caller-identity &> /dev/null; then
                log_error "AWS credentials not configured. Please run 'aws configure'."
                exit 1
            fi
            ;;
    esac
    
    log_info "Prerequisites check passed"
}

# Create secure directory with proper permissions
create_secure_directory() {
    local dir="$1"
    mkdir -p "$dir"
    chmod 700 "$dir"  # Only owner can read/write/execute
}

# Generate TLS certificates
generate_tls_certificates() {
    local cert_dir="$1/certs"
    create_secure_directory "$cert_dir"
    
    if [ ! -f "$cert_dir/server.key" ]; then
        log_info "Generating TLS certificates..."
        
        # Generate private key
        openssl genrsa -out "$cert_dir/server.key" 4096
        
        # Generate certificate signing request
        openssl req -new -key "$cert_dir/server.key" \
            -out "$cert_dir/server.csr" \
            -subj "/C=US/ST=State/L=City/O=Predis/CN=predis-${CUSTOMER_ID}"
        
        # Generate self-signed certificate (for development/testing)
        openssl x509 -req -days 365 \
            -in "$cert_dir/server.csr" \
            -signkey "$cert_dir/server.key" \
            -out "$cert_dir/server.crt"
        
        # Set secure permissions
        chmod 600 "$cert_dir/server.key"
        chmod 644 "$cert_dir/server.crt"
        
        log_info "TLS certificates generated"
    fi
}

# Generate auth tokens file
generate_auth_tokens() {
    local auth_file="$1/auth_tokens.json"
    
    cat > "$auth_file" <<EOF
{
  "tokens": [
    {
      "id": "default",
      "token": "${AUTH_TOKEN}",
      "customer_id": "${CUSTOMER_ID}",
      "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "permissions": ["read", "write", "admin"]
    }
  ]
}
EOF
    
    # Set secure permissions
    chmod 600 "$auth_file"
}

# Generate deployment configuration
generate_config() {
    log_info "Generating deployment configuration for customer: $CUSTOMER_ID"
    
    # Create deployment directory
    DEPLOY_DIR="deployments/${CUSTOMER_ID}"
    create_secure_directory "$DEPLOY_DIR"
    
    # Generate TLS certificates
    generate_tls_certificates "$DEPLOY_DIR"
    
    # Generate auth tokens
    generate_auth_tokens "$DEPLOY_DIR"
    
    # Generate predis.conf
    cat > "$DEPLOY_DIR/predis.conf" <<EOF
# Predis Configuration for Customer: $CUSTOMER_ID
# Generated: $(date)

customer_id: $CUSTOMER_ID
deployment_type: $DEPLOYMENT_TYPE

# Cache Configuration
cache:
  memory_gb: $CACHE_MEMORY_GB
  eviction_policy: lru_with_ml
  persistence_enabled: true
  persistence_path: /var/lib/predis/data

# GPU Configuration
gpu:
  device_id: 0
  memory_pool_size_gb: $CACHE_MEMORY_GB
  kernel_optimization: auto

# MLOps Configuration
mlops:
  enabled: $MLOPS_ENABLED
  data_collection:
    sampling_rate: 0.001
    export_interval_seconds: 60
    export_path: /var/lib/predis/ml_data
  drift_detection:
    enabled: true
    check_interval_minutes: 30
    ks_threshold: 0.05
    psi_threshold: 0.25
  retraining:
    auto_enabled: true
    min_samples: 10000
    performance_threshold: 0.95
    ab_test_traffic_split: 0.1

# Monitoring Configuration
monitoring:
  integration: $MONITORING_INTEGRATION
  metrics_port: 9090
  export_interval_seconds: 10
  
# API Configuration
api:
  redis_port: 6379
  http_port: 8080
  max_connections: 10000
  connection_timeout_ms: 5000

# Security Configuration
security:
  tls_enabled: true
  tls_cert_path: /etc/predis/certs/server.crt
  tls_key_path: /etc/predis/certs/server.key
  auth_enabled: true
  auth_token_file: /etc/predis/auth_tokens.json

# Logging Configuration
logging:
  level: info
  path: /var/log/predis
  max_size_mb: 1000
  max_files: 10
  sensitive_data_masking: true
EOF
    
    # Set secure permissions
    chmod 600 "$DEPLOY_DIR/predis.conf"
    
    log_info "Configuration generated at: $DEPLOY_DIR/predis.conf"
}

# Deploy using Docker (secure version)
deploy_docker() {
    log_info "Deploying Predis using Docker..."
    
    # Create .env file for sensitive data
    cat > "$DEPLOY_DIR/.env" <<EOF
CUSTOMER_ID=${CUSTOMER_ID}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
AUTH_TOKEN=${AUTH_TOKEN}
EOF
    chmod 600 "$DEPLOY_DIR/.env"
    
    # Generate docker-compose.yml
    cat > "$DEPLOY_DIR/docker-compose.yml" <<EOF
version: '3.8'

services:
  predis:
    image: predis/predis:v1.0.0  # Use specific version, not latest
    container_name: predis-\${CUSTOMER_ID}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUSTOMER_ID=\${CUSTOMER_ID}
    volumes:
      - ./predis.conf:/etc/predis/predis.conf:ro
      - ./certs:/etc/predis/certs:ro
      - ./auth_tokens.json:/etc/predis/auth_tokens.json:ro
      - predis-data:/var/lib/predis/data
      - predis-logs:/var/log/predis
      - predis-ml:/var/lib/predis/ml_data
      - predis-models:/var/lib/predis/models
    ports:
      - "127.0.0.1:6379:6379"  # Bind to localhost only
      - "127.0.0.1:8080:8080"
      - "127.0.0.1:9090:9090"
      - "127.0.0.1:8888:8888"
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "/opt/predis/scripts/health_check.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/run

  # Monitoring stack
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: prometheus-\${CUSTOMER_ID}
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "127.0.0.1:9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true

  grafana:
    image: grafana/grafana:9.3.0
    container_name: grafana-\${CUSTOMER_ID}
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SECURITY_DISABLE_GRAVATAR=true
      - GF_SECURITY_COOKIE_SECURE=true
      - GF_SECURITY_COOKIE_SAMESITE=lax
      - GF_SECURITY_CONTENT_SECURITY_POLICY=true
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
    ports:
      - "127.0.0.1:3000:3000"
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    user: "472"  # grafana user

  # ROI Dashboard
  roi-dashboard:
    image: predis/roi-dashboard:v1.0.0
    container_name: roi-dashboard-\${CUSTOMER_ID}
    environment:
      - CUSTOMER_ID=\${CUSTOMER_ID}
      - PREDIS_HOST=predis
      - AUTH_TOKEN=\${AUTH_TOKEN}
    ports:
      - "127.0.0.1:8889:8889"
    depends_on:
      - predis
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp

volumes:
  predis-data:
    driver: local
  predis-logs:
    driver: local
  predis-ml:
    driver: local
  predis-models:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF
    
    # Generate Prometheus configuration
    cat > "$DEPLOY_DIR/prometheus.yml" <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    customer: '${CUSTOMER_ID}'
    environment: 'production'

scrape_configs:
  - job_name: 'predis'
    static_configs:
      - targets: ['predis:9090']
    metrics_path: '/metrics'
    scheme: 'https'
    tls_config:
      insecure_skip_verify: true  # For self-signed certs
EOF
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would deploy with docker-compose in $DEPLOY_DIR"
        return
    fi
    
    # Start services
    cd "$DEPLOY_DIR"
    
    # Pull specific versions
    docker-compose pull
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    
    # More robust health check
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker exec "predis-${CUSTOMER_ID}" redis-cli -a "$AUTH_TOKEN" ping 2>/dev/null | grep -q "PONG"; then
            log_info "Predis is responding"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Predis failed to start within timeout"
        docker-compose logs predis
        exit 1
    fi
    
    log_info "Predis deployed successfully!"
}

# Post-deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Test Redis API with auth
            if docker exec "predis-${CUSTOMER_ID}" redis-cli -a "$AUTH_TOKEN" ping 2>/dev/null | grep -q "PONG"; then
                log_info "✓ Redis API is responding"
            else
                log_error "✗ Redis API is not responding"
                return 1
            fi
            
            # Check metrics endpoint
            if curl -sk "https://localhost:9090/metrics" | grep -q "predis_"; then
                log_info "✓ Metrics endpoint is working"
            else
                log_warn "✗ Metrics endpoint is not responding"
            fi
            
            # Check GPU availability
            if docker exec "predis-${CUSTOMER_ID}" nvidia-smi &>/dev/null; then
                log_info "✓ GPU is accessible"
            else
                log_error "✗ GPU is not accessible"
            fi
            ;;
    esac
}

# Generate secure onboarding documentation
generate_onboarding_docs() {
    log_info "Generating onboarding documentation..."
    
    cat > "$DEPLOY_DIR/ONBOARDING_SECURE.md" <<EOF
# Predis Secure Onboarding Guide for ${CUSTOMER_ID}

## Security Information

### Important Credentials (Store Securely!)
- **Grafana Admin Password**: ${GRAFANA_ADMIN_PASSWORD}
- **Auth Token**: ${AUTH_TOKEN}
- **TLS Certificate**: $DEPLOY_DIR/certs/server.crt

### Security Best Practices
1. Change default passwords immediately
2. Rotate auth tokens regularly (every 90 days)
3. Monitor access logs for suspicious activity
4. Keep deployment files in secure location
5. Use VPN or private network for access

## Access Information

### Redis API (Authenticated)
\`\`\`bash
# With auth token
redis-cli -h localhost -p 6379 -a ${AUTH_TOKEN}

# Using Python
import redis
r = redis.Redis(host='localhost', port=6379, password='${AUTH_TOKEN}', ssl=True)
\`\`\`

### Monitoring Access
- **Grafana**: https://localhost:3000 (admin / [password above])
- **Metrics**: https://localhost:9090/metrics
- **ROI Dashboard**: http://localhost:8889 (requires auth token)

## Security Monitoring

### Check Access Logs
\`\`\`bash
docker exec predis-${CUSTOMER_ID} tail -f /var/log/predis/access.log
\`\`\`

### Monitor Failed Auth Attempts
\`\`\`bash
docker exec predis-${CUSTOMER_ID} grep "AUTH_FAILED" /var/log/predis/security.log
\`\`\`

## Troubleshooting

### Certificate Issues
If you see certificate warnings, you can:
1. Import the self-signed cert to your trust store
2. Generate a proper certificate from a CA
3. Use Let's Encrypt for automatic certificates

### Permission Denied Errors
Check file permissions:
\`\`\`bash
ls -la $DEPLOY_DIR/
# All sensitive files should be 600 (owner read/write only)
\`\`\`

## Support
- **Security Issues**: security@predis.ai (urgent)
- **General Support**: support@predis.ai
- **Documentation**: https://docs.predis.ai/security
EOF
    
    # Set secure permissions on docs
    chmod 600 "$DEPLOY_DIR/ONBOARDING_SECURE.md"
    
    log_info "Secure onboarding guide generated at: $DEPLOY_DIR/ONBOARDING_SECURE.md"
}

# Main execution
main() {
    echo "========================================"
    echo "Predis Secure Customer Deployment"
    echo "========================================"
    echo "Customer ID: $CUSTOMER_ID"
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "========================================"
    
    # Check prerequisites
    check_prerequisites
    
    # Generate configuration
    generate_config
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            log_error "Kubernetes deployment not yet implemented in secure version"
            exit 1
            ;;
        aws)
            log_error "AWS deployment not yet implemented in secure version"
            exit 1
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Validate deployment
    if [ "$DRY_RUN" = false ]; then
        validate_deployment
    fi
    
    # Generate documentation
    generate_onboarding_docs
    
    echo "========================================"
    log_info "Secure deployment completed!"
    log_info "IMPORTANT: Save credentials from: $DEPLOY_DIR/ONBOARDING_SECURE.md"
    echo "========================================"
}

# Run main function
main