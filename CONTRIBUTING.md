# Anime YOLO AI - Contributing Guide

Thank you for considering contributing to Anime YOLO AI! This document provides guidelines for contributing to the project.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards other contributors

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv/conda)
- GPU with CUDA (optional, for faster training)

### Setup Steps
```powershell
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Anime-YOLO-AI.git
cd Anime-YOLO-AI

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### Verify Installation
```powershell
# Run syntax checks
python -m py_compile src/*.py

# Run tests
pytest tests/
```

## 📝 Making Changes

### Branch Naming Convention
- `feature/` - New features (e.g., `feature/add-tracking`)
- `bugfix/` - Bug fixes (e.g., `bugfix/fix-api-latency`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/optimize-data-pipeline`)

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Build process or auxiliary tool changes

**Example**:
```
feat(api): add batch prediction endpoint

- Implement POST /batch-predict for multiple images
- Add async processing with Celery
- Update API documentation

Closes #42
```

### Code Organization
```
Anime-YOLO-AI/
├── src/              # Core ML scripts
├── api/              # FastAPI application
├── tests/            # Unit and integration tests
├── docs/             # Additional documentation
└── .github/          # CI/CD workflows
```

## 🧪 Testing

### Running Tests
```powershell
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_pipeline.py
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names

**Example**:
```python
# tests/test_data_pipeline.py
import pytest
from src.validate_images import is_valid_image

def test_is_valid_image_with_jpg():
    """Test that valid JPG images pass validation"""
    assert is_valid_image('test_data/valid.jpg') == True

def test_is_valid_image_with_gif():
    """Test that GIF images are rejected"""
    assert is_valid_image('test_data/animated.gif') == False
```

## 🔄 Pull Request Process

### Before Submitting
1. **Update Documentation**: Ensure README and docstrings are current
2. **Run Tests**: All tests must pass
3. **Code Quality**: Run linters (black, flake8)
4. **Add Tests**: Include tests for new features

### Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow convention
- [ ] PR description clearly explains changes

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] CI/CD pipeline passes

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code sections
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process
1. Maintainers review within 3-5 business days
2. Address feedback and update PR
3. Once approved, maintainer will merge

## 📐 Coding Standards

### Python Style Guide
- Follow PEP 8
- Use Black for formatting: `black src/ api/`
- Line length: 100 characters
- Use type hints where possible

**Example**:
```python
def download_image(url: str, output_path: str, retries: int = 3) -> bool:
    """
    Download image from URL with retry logic.
    
    Args:
        url: Image URL to download
        output_path: Local path to save image
        retries: Number of retry attempts (default: 3)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    # Implementation
    pass
```

### Documentation Standards
- All functions must have docstrings
- Use Google-style docstrings
- Include type hints in function signatures
- Add inline comments for complex logic

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Local imports
from src.utils import load_config
from src.data_prep import validate_dataset
```

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [Windows/Linux/macOS]
- Python version: 3.11.x
- GPU: [NVIDIA RTX 3080 / CPU only]

**Logs**
Paste relevant logs or error messages
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution**
Clear description of desired solution

**Alternatives considered**
Alternative solutions or features

**Additional context**
Screenshots, mockups, or examples
```

## 📞 Contact

- **GitHub Issues**: [Create an issue](https://github.com/username/Anime-YOLO-AI/issues)
- **Email**: your.email@example.com
- **Discord**: [Project Discord Server](#)

## 🎓 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Python Testing with pytest](https://docs.pytest.org/)

---

Thank you for contributing to Anime YOLO AI! 🚀
