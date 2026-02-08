# Contributing to LlamaCPP Studio

**Created by**: Damian Sromek

Thank you for your interest in contributing to LlamaCPP Studio! This guide will help you get started with making contributions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Project Guidelines](#project-guidelines)

## 🤝 Code of Conduct

This project adheres to a code of conduct. Please read and follow these guidelines to help create a positive and inclusive community environment.

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone. We expect all contributors to:

- Treat everyone with respect and professionalism
- Be open and constructive in discussions
- Welcome new ideas and perspectives
- Refrain from harassment or discrimination

### Code of Conduct Violations

If you witness or experience any violations of this code of conduct, please report them to the project maintainers through GitHub issues.

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Basic Git Knowledge**: Understanding of version control
2. **Forked Repository**: Created your own fork of the project
3. **Python Environment**: Setup with Python 3.9+
4. **Development Tools**: Git, IDE (VS Code, PyCharm, etc.)

### Setting Up Your Development Environment

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/yourusername/paq_llamacpp_studio.git
cd paq_llamacpp_studio

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Setup development environment
./scripts/setup.sh

# 5. Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov httpx
```

## 🔧 Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/paq_llamacpp_studio.git
```

### 2. Set Up Your Branch

```bash
# Checkout main branch
git checkout main

# Pull latest changes
git pull origin main

# Create your feature branch
git checkout -b feature/my-cool-feature
```

### 3. Make Changes

- Make your changes in your feature branch
- Follow the coding standards below
- Keep changes focused and minimal

### 4. Test Your Changes

```bash
# Run existing tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/unit/test_module.py

# Run with coverage
python3 -m pytest tests/ --cov=tools --cov-report=html
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Create a descriptive commit message
git commit -m "feat: add new feature that does something"

# Push to your fork
git push origin feature/your-feature-name
```

### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill in the PR description template
5. Submit for review

## 📝 Coding Standards

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines:

- Use 4 spaces per indent level
- Limit lines to 120 characters
- Use meaningful variable and function names
- Write docstrings for all functions and classes
- Use type hints for better code documentation

### Commit Message Format

Follow conventional commits specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(benchmark): add new performance metrics
fix(cli): resolve command line argument parsing issue
docs(readme): update installation instructions
test(utils): add unit tests for utility functions
```

### Code Structure

```python
# Import standard library first
import os
import sys

# Then import third-party libraries
import requests
from typing import List, Optional

# Finally import local modules
from tools.llama_bench import cli
from tools.llama_bench.utils import helpers

# Constants and configuration
DEFAULT_PORT = 11433
MAX_RETRIES = 3

# Classes and functions
class MyClass:
    """Class docstring"""
    
    def __init__(self, config: dict):
        """Constructor docstring"""
        self.config = config

    def my_method(self, value: int) -> Optional[dict]:
        """Method docstring with details"""
        if value < 0:
            return None
        return {"result": value}

# Main execution block
if __name__ == "__main__":
    # Main code
    pass
```

## 📊 Pull Request Process

### PR Template

Please include the following information in your PR:

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added or updated
- [ ] Documentation updated
- [ ] Changes are minimal and focused

## Related Issues
Fixes #123

## How to Test
Describe how to test these changes

## Screenshots
(If applicable) Add screenshots showing the changes
```

### Review Process

1. **Automated Checks**: Pass all CI/CD checks
2. **Code Review**: Get feedback from maintainers
3. **Changes**: Address review comments
4. **Approval**: Get approval from maintainers
5. **Merge**: Your PR is merged

### PR Naming Convention

```bash
# Follow format
type(scope): description
```

**Examples:**
```bash
feat(benchmark): add GPU performance metrics
fix(cli): resolve parameter parsing issue
docs(readme): update installation guide
```

## 🎯 Project Guidelines

### Areas for Contribution

1. **Feature Development**: New tools, capabilities, and improvements
2. **Bug Fixes**: Resolving issues reported by users
3. **Documentation**: Improving guides and examples
4. **Testing**: Adding test cases and coverage
5. **Performance**: Optimizing code and tooling
6. **User Experience**: UI/UX improvements

### What We're Looking For

- High-quality, well-tested code
- Clear documentation
- Follows existing code patterns
- Minimal, focused changes
- Good error handling
- Performance-conscious code

### What to Avoid

- Large, sweeping changes
- Breaking changes without discussion
- Incomplete implementations
- Poorly documented code
- Performance regressions

## 🐛 Bug Reporting

Before contributing, check existing issues:

1. Search for similar bugs in GitHub Issues
2. Check if the issue has been resolved
3. If no match, create a new issue with details

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- GPU: [e.g., NVIDIA RTX 3080]

## Additional Context
Any other relevant information or error messages
```

## 📚 Additional Resources

- [Project Documentation](../docs/)
- [Quick Start Guide](../docs/QUICKSTART.md)
- [API Documentation](../tools/)
- [Testing Guide](../docs/BENCHMARKING.md)

## 🙏 Thank You!

Contributions like yours help make LlamaCPP Studio better for everyone. We appreciate your time and effort in improving this project!

---

**Remember**: The best contributions are those that help other developers. Keep it simple, keep it focused, and keep it useful.
