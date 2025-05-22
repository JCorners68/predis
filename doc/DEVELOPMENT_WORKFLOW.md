# Predis Development Workflow & Standards

This document outlines the development workflow, coding standards, and collaboration practices for the Predis project.

## Git Workflow

### Branching Strategy

**Main Branches:**
- `main` - Production-ready code, always deployable
- `develop` - Integration branch for features, epic development

**Feature Branches:**
- `feature/epic-X-story-Y-description` - Individual story implementation
- `feature/gpu-memory-optimization` - Major feature work
- `hotfix/critical-bug-fix` - Critical production fixes

**Example Branch Names:**
```
feature/epic-1-story-1-gpu-memory-manager
feature/epic-1-story-2-cuda-hash-table
hotfix/memory-leak-cache-manager
```

### Development Process

1. **Start New Work:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/epic-X-story-Y-description
   ```

2. **Regular Development:**
   ```bash
   # Make changes
   git add .
   git commit -m "Add GPU memory allocation for cache entries
   
   - Implement CUDA memory management in MemoryManager
   - Add error handling for VRAM allocation failures
   - Update cache size tracking and statistics
   
   🤖 Generated with Claude Code"
   ```

3. **Push and Create PR:**
   ```bash
   git push -u origin feature/epic-X-story-Y-description
   # Create pull request via GitHub CLI or web interface
   ```

4. **Merge Process:**
   - **Squash merge** for feature branches to main
   - **Merge commit** for hotfixes to preserve history
   - Delete feature branch after merge

### Commit Message Standards

**Format:**
```
<type>: <short description>

<detailed description>
- <specific change 1>
- <specific change 2>
- <specific change 3>

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix  
- `perf:` - Performance improvement
- `refactor:` - Code restructuring
- `test:` - Adding/updating tests
- `docs:` - Documentation changes
- `build:` - Build system changes

## Code Review Process

### Pull Request Requirements

**Before Creating PR:**
- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated for new features
- [ ] Performance impact assessed
- [ ] GPU memory usage validated

**PR Template:**
```markdown
## Summary
Brief description of changes and why they were made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks run
- [ ] GPU functionality validated

## Performance Impact
- Baseline: X ops/sec
- After changes: Y ops/sec
- Memory usage change: +/- Z MB

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No debugging code left in
- [ ] Error handling added
```

### Review Guidelines

**Reviewer Checklist:**
- [ ] **Functionality** - Does the code do what it claims?
- [ ] **Performance** - Any GPU/memory performance regressions?
- [ ] **Error Handling** - Proper CUDA error checking and recovery?
- [ ] **Testing** - Adequate test coverage for new code?
- [ ] **Style** - Follows project coding standards?
- [ ] **Documentation** - Clear comments and updated docs?

**Review Timeline:**
- **Small changes** (<100 lines): 24 hours
- **Medium changes** (100-500 lines): 48 hours  
- **Large changes** (>500 lines): 72 hours

## Testing Strategy

### Test Coverage Requirements

**Minimum Coverage:**
- **Unit Tests**: 80% line coverage
- **Integration Tests**: All major workflows
- **Performance Tests**: Baseline measurements for all operations

**Test Categories:**

1. **Unit Tests** (`tests/unit/`)
   - Individual class/function testing
   - Mock dependencies for isolation
   - Fast execution (<1 second total)

2. **Integration Tests** (`tests/integration/`)
   - End-to-end workflow testing
   - GPU memory allocation/deallocation
   - Client API to cache core integration

3. **Performance Tests** (`tests/performance/`)
   - Benchmarking against Redis baseline
   - Memory usage validation
   - Throughput and latency measurements

4. **GPU Tests** (`tests/gpu/`)
   - CUDA kernel functionality
   - GPU memory management
   - Error handling for GPU failures

### Test Execution

**Local Development:**
```bash
# Run all tests
make test

# Run specific test suites
make unit-tests
make integration-tests
make performance-tests

# Run with coverage
make test-coverage
```

**Docker Environment:**
```bash
# GPU-specific tests
./dev_scripts/start_dev.sh
docker-compose -f docker-compose.dev.yml exec predis-dev make gpu-tests
```

## Release Process

### Version Numbering

**Semantic Versioning (MAJOR.MINOR.PATCH):**
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Pre-release Tags:**
- `v0.1.0-alpha.1` - Alpha releases for internal testing
- `v0.1.0-beta.1` - Beta releases for external validation
- `v0.1.0-rc.1` - Release candidates

### Release Checklist

**Pre-Release:**
- [ ] All epic stories completed
- [ ] Performance benchmarks meet targets
- [ ] Documentation updated
- [ ] Security review completed
- [ ] GPU compatibility validated

**Release Process:**
1. Create release branch: `release/v0.1.0`
2. Update version numbers and changelog
3. Run full test suite including performance tests
4. Create GitHub release with binaries
5. Update documentation and demo materials

**Post-Release:**
- [ ] Monitor performance metrics
- [ ] Gather user feedback
- [ ] Plan next iteration based on learnings

## Development Environment Setup

### Required Tools

**Development:**
- Docker Desktop with GPU support
- Git with LFS for large files
- clang-format for code formatting
- CMake 3.18+ for building

**Code Quality:**
- pre-commit hooks for automated checks
- clang-tidy for static analysis
- valgrind for memory leak detection
- nvidia-nsight for GPU profiling

### IDE Configuration

**Recommended Settings:**
- Tab size: 4 spaces
- Line endings: LF (Unix style)
- Trailing whitespace: Remove automatically
- clang-format integration enabled

**VS Code Extensions:**
- C/C++ Extension Pack
- CMake Tools
- GitLens
- Docker Extension

## Collaboration Guidelines

### Communication

**Channels:**
- **Google Chat**: General discussions and quick questions
- **GitHub Issues**: Bug reports and feature requests
- **Pull Request Reviews**: Code-specific discussions
- **Documentation**: Architecture decisions and long-term planning

### Documentation Standards

**Always Update:**
- README.md for setup changes
- CLAUDE.md for new efficiency patterns
- Architecture docs for design decisions
- Performance baselines for optimization work

**Code Documentation:**
- Header comments for all public APIs
- Inline comments for complex GPU operations
- Performance notes for optimization choices
- Error handling documentation

### Performance Culture

**Always Measure:**
- Baseline performance before changes
- Memory usage impact
- GPU utilization efficiency
- Comparison against Redis when relevant

**Performance Reviews:**
- Include benchmark results in PRs
- Document performance trade-offs
- Validate GPU memory usage
- Test on target hardware (RTX 5080)

This workflow ensures high code quality, efficient collaboration, and maintainable GPU-accelerated performance for the Predis project.