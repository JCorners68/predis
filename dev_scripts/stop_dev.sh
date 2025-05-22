#!/bin/bash

# Predis Development Environment Stop Script

set -e

echo "🛑 Stopping Predis Development Environment..."

# Stop all services
docker-compose -f docker-compose.dev.yml down

# Optionally clean up volumes
read -p "Remove persistent volumes (build cache, logs)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning up volumes..."
    docker-compose -f docker-compose.dev.yml down -v
    docker volume prune -f
fi

echo "✅ Development environment stopped"