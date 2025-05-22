# Contributing to Predis

Welcome to the Predis project! We appreciate your interest in contributing to our GPU-accelerated key-value cache with predictive prefetching capabilities.

## Development Setup

### Prerequisites
- WSL2 with Ubuntu 22.04 LTS
- NVIDIA RTX 5080 (or compatible GPU with 16GB+ VRAM)
- Docker with NVIDIA container runtime
- CUDA 12.x development toolkit
- Python 3.10+

### Environment Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd predis
   ```

2. Set up development environment:
   ```bash
   ./scripts/setup/setup_dev_env.sh
   ```

3. Verify GPU access:
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
   ```

## Development Workflow

### Git Workflow
We use Git Flow branching model:
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature development branches
- `hotfix/*` - Critical bug fixes

### Creating a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes
1. Make your changes following our coding standards
2. Add tests for new functionality
3. Run the test suite: `make test`
4. Run linting: `make lint`
5. Ensure GPU tests pass: `make test-gpu`

### Submitting Changes
1. Commit your changes with descriptive messages
2. Push to your feature branch
3. Create a pull request to `develop` branch
4. Ensure all CI checks pass
5. Request review from maintainers

## Coding Standards

### C++/CUDA Code
- Follow Google C++ Style Guide
- Use clang-format for automatic formatting
- Include comprehensive comments for GPU kernels
- Memory management must be explicit and leak-free
- All CUDA code must include error checking

### Python Code
- Follow PEP 8 style guide
- Use black for automatic formatting
- Include type hints for all functions
- Docstrings required for all public functions
- Maximum line length: 88 characters

### Mojo Code
- Follow emerging Mojo best practices
- Optimize for GPU execution where applicable
- Document performance-critical sections
- Use consistent naming conventions

## Testing

### Unit Tests
- C++: Google Test framework
- Python: pytest
- GPU tests: CUDA-specific test utilities
- Target: >90% code coverage

### Integration Tests
- End-to-end cache operations
- Performance benchmarking
- Redis comparison tests
- ML model validation

### Performance Tests
- Latency measurements
- Throughput benchmarks
- Memory usage validation
- GPU utilization metrics

## Documentation

### Code Documentation
- Doxygen for C++/CUDA code
- Sphinx for Python documentation
- Inline comments for complex algorithms
- Performance annotations for critical paths

### User Documentation
- API reference updates
- Architecture documentation
- Performance tuning guides
- Troubleshooting guides

## Performance Requirements

### Benchmarking
- All performance claims must be reproducible
- Include statistical significance testing
- Compare against Redis baseline
- Document hardware configuration

### Optimization Guidelines
- GPU memory bandwidth utilization >80%
- CUDA kernel occupancy >75%
- Memory transfer overhead <10%
- Cache hit rate improvement >20% with ML

## Code Review Process

### Review Criteria
- Code quality and style compliance
- Test coverage and quality
- Performance impact assessment
- Documentation completeness
- Security considerations

### GPU-Specific Reviews
- CUDA kernel efficiency
- VRAM usage optimization
- Memory coalescing patterns
- Error handling for GPU operations

## Issue Reporting

### Bug Reports
Include:
- GPU hardware specifications
- CUDA/driver versions
- Reproduction steps
- Expected vs actual behavior
- Performance impact

### Feature Requests
Include:
- Use case description
- Performance requirements
- API design considerations
- Implementation complexity estimate

## Security Guidelines

### GPU Security
- No credential storage in GPU memory
- Secure memory clearing after use
- Input validation for all GPU operations
- Protection against memory corruption

### General Security
- No hardcoded secrets
- Secure communication protocols
- Input sanitization
- Dependency vulnerability scanning

## License Headers

All source files must include the Apache 2.0 license header:

### C++/CUDA Files
```cpp
/*
 * Copyright 2025 Predis Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

### Python Files
```python
# Copyright 2025 Predis Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Mojo Files
```mojo
# Copyright 2025 Predis Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Community

### Communication
- GitHub Issues for bug reports and features
- GitHub Discussions for general questions
- Code reviews for technical discussions

### Recognition
Contributors will be acknowledged in:
- CHANGELOG.md
- Release notes
- Project documentation
- Git commit co-authorship

## License

By contributing to Predis, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search GitHub Issues
3. Create a new issue with the "question" label

Thank you for contributing to Predis!