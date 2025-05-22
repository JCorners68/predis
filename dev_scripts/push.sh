#!/bin/bash

# Automatic Git Push Script for Solo Developer
# Usage: ./push.sh "commit message"
# Author: Solo Developer Script for Predis Project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if commit message is provided
if [ $# -eq 0 ]; then
    print_error "No commit message provided!"
    echo "Usage: $0 \"Your commit message\""
    echo "Example: $0 \"Add Epic 0 dependency chart and push script\""
    exit 1
fi

COMMIT_MESSAGE="$1"

print_status "Starting automated push process..."
print_status "Commit message: \"$COMMIT_MESSAGE\""

# Navigate to project root (in case script is run from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_status "Working directory: $(pwd)"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository!"
    exit 1
fi

# Check git status
print_status "Checking git status..."
if ! git status > /dev/null 2>&1; then
    print_error "Git status check failed!"
    exit 1
fi

# Show current status
echo ""
print_status "Current git status:"
git status --short

# Check if there are any changes to commit
if git diff --quiet && git diff --cached --quiet; then
    print_warning "No changes to commit!"
    print_status "Repository is clean."
    exit 0
fi

# Add all changes
print_status "Adding all changes..."
git add .

# Show what will be committed
echo ""
print_status "Changes to be committed:"
git diff --cached --name-status

# Create commit
print_status "Creating commit..."
git commit -m "$COMMIT_MESSAGE

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
print_status "Current branch: $CURRENT_BRANCH"

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    print_error "No remote 'origin' configured!"
    print_status "Configure remote with: git remote add origin <your-repo-url>"
    exit 1
fi

# Check if current branch tracks a remote branch
if ! git rev-parse --abbrev-ref @{u} > /dev/null 2>&1; then
    print_warning "Current branch '$CURRENT_BRANCH' is not tracking a remote branch."
    print_status "Setting upstream to origin/$CURRENT_BRANCH..."
    git push -u origin "$CURRENT_BRANCH"
else
    # Push to remote
    print_status "Pushing to remote repository..."
    git push
fi

# Verify push was successful
if [ $? -eq 0 ]; then
    print_success "Successfully pushed to remote repository!"
    print_success "Branch: $CURRENT_BRANCH"
    print_success "Commit: $(git rev-parse --short HEAD)"
    
    # Show remote URL (without credentials)
    REMOTE_URL=$(git remote get-url origin | sed 's/https:\/\/[^@]*@/https:\/\//')
    print_status "Remote: $REMOTE_URL"
else
    print_error "Push failed!"
    exit 1
fi

echo ""
print_success "🚀 Push completed successfully!"