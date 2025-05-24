#!/bin/bash

# Predis Customer Deployment Script
# Automates deployment for pilot customers with monitoring and validation

set -e

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --customer-id)
            CUSTOMER_ID="$2"
            shift 2
            ;;
        --deployment-type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --cache-memory)
            CACHE_MEMORY_GB="$2"
            shift 2
            ;;
        --no-mlops)
            MLOPS_ENABLED="false"
            shift
            ;;
        --monitoring)
            MONITORING_INTEGRATION="$2"
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
            echo "  --monitoring        prometheus|datadog|cloudwatch (default: prometheus)"
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

# Generate deployment configuration
generate_config() {
    log_info "Generating deployment configuration for customer: $CUSTOMER_ID"
    
    # Create deployment directory
    DEPLOY_DIR="deployments/${CUSTOMER_ID}"
    mkdir -p "$DEPLOY_DIR"
    
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
EOF
    
    log_info "Configuration generated at: $DEPLOY_DIR/predis.conf"
}

# Deploy using Docker
deploy_docker() {
    log_info "Deploying Predis using Docker..."
    
    # Generate docker-compose.yml
    cat > "$DEPLOY_DIR/docker-compose.yml" <<EOF
version: '3.8'

services:
  predis:
    image: predis/predis:latest
    container_name: predis-${CUSTOMER_ID}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUSTOMER_ID=${CUSTOMER_ID}
    volumes:
      - ./predis.conf:/etc/predis/predis.conf:ro
      - predis-data:/var/lib/predis/data
      - predis-logs:/var/log/predis
      - predis-ml:/var/lib/predis/ml_data
      - predis-models:/var/lib/predis/models
    ports:
      - "6379:6379"  # Redis API
      - "8080:8080"  # HTTP API
      - "9090:9090"  # Metrics
      - "8888:8888"  # Dashboard
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "predis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Monitoring stack
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-${CUSTOMER_ID}
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-${CUSTOMER_ID}
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=predis123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    restart: unless-stopped

  # ROI Dashboard
  roi-dashboard:
    image: predis/roi-dashboard:latest
    container_name: roi-dashboard-${CUSTOMER_ID}
    environment:
      - CUSTOMER_ID=${CUSTOMER_ID}
      - PREDIS_HOST=predis
    ports:
      - "8889:8889"
    depends_on:
      - predis
    restart: unless-stopped

volumes:
  predis-data:
  predis-logs:
  predis-ml:
  predis-models:
  prometheus-data:
  grafana-data:
EOF
    
    # Generate Prometheus configuration
    cat > "$DEPLOY_DIR/prometheus.yml" <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'predis'
    static_configs:
      - targets: ['predis:9090']
    metrics_path: '/metrics'
EOF
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would deploy with docker-compose in $DEPLOY_DIR"
        return
    fi
    
    # Start services
    cd "$DEPLOY_DIR"
    docker-compose pull
    docker-compose up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "healthy"; then
        log_info "Predis deployed successfully!"
    else
        log_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Deploy using Kubernetes
deploy_kubernetes() {
    log_info "Deploying Predis using Kubernetes..."
    
    # Create namespace
    kubectl create namespace "predis-${CUSTOMER_ID}" --dry-run=client -o yaml > "$DEPLOY_DIR/namespace.yaml"
    
    # Generate Kubernetes manifests
    cat > "$DEPLOY_DIR/predis-deployment.yaml" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predis
  namespace: predis-${CUSTOMER_ID}
  labels:
    app: predis
    customer: ${CUSTOMER_ID}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: predis
  template:
    metadata:
      labels:
        app: predis
    spec:
      containers:
      - name: predis
        image: predis/predis:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CUSTOMER_ID
          value: "${CUSTOMER_ID}"
        volumeMounts:
        - name: config
          mountPath: /etc/predis
        - name: data
          mountPath: /var/lib/predis/data
        - name: logs
          mountPath: /var/log/predis
        ports:
        - containerPort: 6379
          name: redis
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: predis-config
      - name: data
        persistentVolumeClaim:
          claimName: predis-data
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: predis
  namespace: predis-${CUSTOMER_ID}
spec:
  selector:
    app: predis
  ports:
  - name: redis
    port: 6379
    targetPort: 6379
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: predis-data
  namespace: predis-${CUSTOMER_ID}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
EOF
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would deploy to Kubernetes namespace: predis-${CUSTOMER_ID}"
        return
    fi
    
    # Apply manifests
    kubectl apply -f "$DEPLOY_DIR/namespace.yaml"
    kubectl create configmap predis-config -n "predis-${CUSTOMER_ID}" --from-file="$DEPLOY_DIR/predis.conf"
    kubectl apply -f "$DEPLOY_DIR/predis-deployment.yaml"
    
    # Wait for deployment
    kubectl wait --for=condition=available --timeout=300s deployment/predis -n "predis-${CUSTOMER_ID}"
    
    log_info "Predis deployed to Kubernetes successfully!"
}

# Deploy using AWS CloudFormation
deploy_aws() {
    log_info "Deploying Predis using AWS CloudFormation..."
    
    STACK_NAME="predis-${CUSTOMER_ID}"
    TEMPLATE_FILE="$(dirname "$0")/../cloud_marketplace/aws/cloudformation_template.yaml"
    
    if [ ! -f "$TEMPLATE_FILE" ]; then
        log_error "CloudFormation template not found: $TEMPLATE_FILE"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would create CloudFormation stack: $STACK_NAME"
        return
    fi
    
    # Create stack
    aws cloudformation create-stack \
        --stack-name "$STACK_NAME" \
        --template-body "file://$TEMPLATE_FILE" \
        --parameters \
            ParameterKey=CustomerID,ParameterValue="$CUSTOMER_ID" \
            ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE" \
            ParameterKey=CacheMemoryGB,ParameterValue="$CACHE_MEMORY_GB" \
            ParameterKey=MLOpsEnabled,ParameterValue="$MLOPS_ENABLED" \
            ParameterKey=MonitoringIntegration,ParameterValue="$MONITORING_INTEGRATION" \
        --capabilities CAPABILITY_IAM
    
    # Wait for stack creation
    log_info "Waiting for CloudFormation stack creation..."
    aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
    
    # Get outputs
    OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs')
    echo "$OUTPUTS" > "$DEPLOY_DIR/aws-outputs.json"
    
    log_info "Predis deployed to AWS successfully!"
    log_info "Stack outputs saved to: $DEPLOY_DIR/aws-outputs.json"
}

# Post-deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Test Redis API
            if docker exec "predis-${CUSTOMER_ID}" redis-cli ping | grep -q "PONG"; then
                log_info "✓ Redis API is responding"
            else
                log_error "✗ Redis API is not responding"
                return 1
            fi
            
            # Check metrics endpoint
            if curl -s "http://localhost:9090/metrics" | grep -q "predis_"; then
                log_info "✓ Metrics endpoint is working"
            else
                log_warn "✗ Metrics endpoint is not responding"
            fi
            
            # Check dashboard
            if curl -s "http://localhost:8888" | grep -q "Predis"; then
                log_info "✓ Dashboard is accessible"
            else
                log_warn "✗ Dashboard is not responding"
            fi
            ;;
        kubernetes)
            # Get service endpoint
            ENDPOINT=$(kubectl get svc predis -n "predis-${CUSTOMER_ID}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            if [ -n "$ENDPOINT" ]; then
                log_info "✓ Service endpoint: $ENDPOINT"
            else
                log_error "✗ Service endpoint not available yet"
            fi
            ;;
        aws)
            # Check CloudFormation stack status
            STATUS=$(aws cloudformation describe-stacks --stack-name "predis-${CUSTOMER_ID}" --query 'Stacks[0].StackStatus' --output text)
            if [ "$STATUS" = "CREATE_COMPLETE" ]; then
                log_info "✓ CloudFormation stack created successfully"
            else
                log_error "✗ CloudFormation stack status: $STATUS"
            fi
            ;;
    esac
}

# Generate onboarding documentation
generate_onboarding_docs() {
    log_info "Generating onboarding documentation..."
    
    cat > "$DEPLOY_DIR/ONBOARDING.md" <<EOF
# Predis Onboarding Guide for ${CUSTOMER_ID}

## Deployment Information
- **Customer ID**: ${CUSTOMER_ID}
- **Deployment Type**: ${DEPLOYMENT_TYPE}
- **Deployment Date**: $(date)
- **Cache Memory**: ${CACHE_MEMORY_GB}GB
- **MLOps**: ${MLOPS_ENABLED}

## Access Information

### Redis API
- **Port**: 6379
- **Protocol**: Redis protocol (compatible with existing Redis clients)
- **Connection String**: \`redis://localhost:6379\` (adjust host for your deployment)

### HTTP API
- **Port**: 8080
- **Endpoint**: \`http://localhost:8080/api/v1\`
- **Documentation**: \`http://localhost:8080/docs\`

### Monitoring
- **Metrics**: \`http://localhost:9090/metrics\` (Prometheus format)
- **Dashboard**: \`http://localhost:8888\`
- **Grafana**: \`http://localhost:3000\` (admin/predis123)
- **ROI Dashboard**: \`http://localhost:8889\`

## Quick Start

### 1. Test Connection
\`\`\`bash
# Using redis-cli
redis-cli -h localhost -p 6379 ping

# Using Python
import redis
r = redis.Redis(host='localhost', port=6379)
r.ping()
\`\`\`

### 2. Basic Operations
\`\`\`python
# Set a value
r.set('key1', 'value1')

# Get a value
value = r.get('key1')

# Batch operations (leverages GPU)
pipe = r.pipeline()
for i in range(1000):
    pipe.set(f'key{i}', f'value{i}')
pipe.execute()
\`\`\`

### 3. Enable ML Prefetching
\`\`\`python
# Configure prefetching
r.execute_command('PREDIS.CONFIG', 'SET', 'prefetch.enabled', 'true')
r.execute_command('PREDIS.CONFIG', 'SET', 'prefetch.confidence_threshold', '0.7')

# Hint related keys for better predictions
r.execute_command('PREDIS.HINT', 'user:123:*', 'user:123:profile', 'user:123:settings')
\`\`\`

## Performance Tuning

### Optimize for Your Workload
1. **ML Training Workload**: 
   - Enable aggressive prefetching
   - Increase batch size for better GPU utilization
   
2. **High-Frequency Trading**:
   - Minimize latency with direct GPU memory access
   - Disable prefetching for deterministic latency

3. **Gaming/Streaming**:
   - Enable pattern-based prefetching
   - Configure larger cache memory allocation

## Monitoring and Alerts

### Key Metrics to Monitor
- **Cache Hit Rate**: Target >80% with ML prefetching
- **Average Latency**: Should be <1ms for cached items
- **GPU Utilization**: Higher is better (target >70%)
- **ML Model Accuracy**: Monitor drift and retrain as needed

### Setting Up Alerts
Example Prometheus alert rules are provided in \`$DEPLOY_DIR/alerts.yml\`

## Support

- **Documentation**: https://docs.predis.ai
- **Support Email**: support@predis.ai
- **Slack Channel**: [Join our Slack](#)

## Next Steps

1. Run the included benchmark suite to establish baseline performance
2. Configure monitoring dashboards for your specific metrics
3. Schedule a review meeting with our team in 1 week
4. Provide feedback on initial performance results
EOF
    
    log_info "Onboarding guide generated at: $DEPLOY_DIR/ONBOARDING.md"
}

# Main execution
main() {
    echo "========================================"
    echo "Predis Customer Deployment Script"
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
            deploy_kubernetes
            ;;
        aws)
            deploy_aws
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
    log_info "Deployment completed successfully!"
    log_info "Onboarding guide: $DEPLOY_DIR/ONBOARDING.md"
    echo "========================================"
}

# Run main function
main