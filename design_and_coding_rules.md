# Movement Analysis Project - Development Guide

## Table of Contents

1. Introduction
2. Project Overview
3. Development Environment
4. Code Quality Tools
5. Component-Specific Guidelines
   - ML Development
   - Backend Development
   - Frontend Development
6. Workflows and CI/CD
7. Deployment and Maintenance

## 1. Introduction

This guide outlines development standards and practices for our Movement Analysis Project. The project combines machine learning models, a FastAPI backend service, and a React frontend to provide movement analysis capabilities.

### Why This Guide

- Ensure consistent development practices
- Streamline onboarding for new team members
- Maintain code quality across components
- Document key decisions and practices

### Note on Simplicity

We prioritize simplicity and maintainability in our tooling choices. For example, we use requirements.txt instead of more complex tools like Poetry because:

- Simple setup and usage
- Widely understood by Python developers
- No additional learning curve
- Easy to troubleshoot and modify
- Direct integration with pip

## 2. Project Overview

### Structure

```
project/
├── ML/
│   ├── requirements.txt
│   ├── models/           # Trained models by type
│   ├── training/         # Training scripts
│   ├── utils/           # Shared utilities
│   └── tests/
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py      # FastAPI application
│   │   └── model_loader.py
│   └── tests/
└── frontend/
    ├── package.json
    ├── src/
    └── tests/
```

### Components

1. **ML**

   - Purpose: Model training and evaluation
   - Key files: Trained models and training scripts
   - Output: Serialized models for backend use

2. **Backend**

   - Purpose: Serve ML models via API
   - Key files: FastAPI app and model loader
   - Output: REST API endpoints

3. **Frontend**
   - Purpose: User interface
   - Key files: React components
   - Output: Web interface

## 3. Development Environment

### Python Setup (ML and Backend)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

Key Python dependencies:

```
# ML/requirements.txt and backend/requirements.txt
black==23.12.1     # Code formatter
flake8==7.0.0      # Code quality checker
isort==5.13.2      # Import organizer
pytest==7.4.4      # Testing

# Additional ML dependencies
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.4.0

# Additional Backend dependencies
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
```

### Frontend Setup

```bash
cd frontend
npm install
```

Key frontend dependencies:

```json
{
  "dependencies": {
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "axios": "1.6.2"
  },
  "devDependencies": {
    "eslint": "8.56.0",
    "prettier": "3.1.1",
    "jest": "29.7.0"
  }
}
```

## 4. Code Quality Tools

### Python Code Quality

Used in both ML and Backend:

1. **Black (Formatter)**

   ```bash
   black .  # Formats all Python files
   ```

   Example:

   ```python
   # Before
   def function   (x,y=42): return x+y

   # After
   def function(x, y=42):
       return x + y
   ```

2. **isort (Import Organizer)**

   ```bash
   isort .  # Organizes imports
   ```

   Example:

   ```python
   # Before
   import pandas
   from fastapi import FastAPI
   import numpy

   # After
   import numpy
   import pandas
   from fastapi import FastAPI
   ```

3. **flake8 (Code Quality)**
   ```bash
   flake8  # Checks code quality
   ```
   Checks for:
   - Line length
   - Syntax errors
   - Style issues
   - Common mistakes

### Frontend Code Quality

1. **Prettier**

   ```bash
   npm run format
   ```

2. **ESLint**
   ```bash
   npm run lint
   ```

## 5. Component-Specific Guidelines

### ML Development

1. **Model Organization**

   - Group models by type in models/
   - Include metadata with each model
   - Version models consistently

2. **Development Steps**

   ```bash
   cd ML
   source venv/bin/activate

   # Quality checks
   black .
   isort .
   flake8

   # Run tests
   pytest
   ```

### Backend Development

1. **API Development**

   - Follow RESTful principles
   - Document endpoints
   - Handle errors consistently

2. **Development Steps**

   ```bash
   cd backend
   source venv/bin/activate

   # Quality checks
   black .
   isort .
   flake8

   # Run server
   uvicorn app.main:app --reload

   # Run tests
   pytest
   ```

### Frontend Development

1. **Component Development**

   - Keep components focused
   - Handle loading and error states
   - Write meaningful tests

2. **Development Steps**

   ```bash
   cd frontend

   # Start development server
   npm start

   # Quality checks
   npm run format
   npm run lint

   # Run tests
   npm test
   ```

## 6. Development Workflows

### Git Workflow

1. **Branch Strategy**

   - Main branch: `main`
   - Feature branches: `feature/description`
   - Bug fixes: `fix/description`

2. **Commit Messages**

   ```
   Feature: Add new model type
   Fix: Update API response format
   Update: Improve error handling
   ```

3. **Pull Request Process**
   - Create feature branch
   - Make changes
   - Run tests and quality checks
   - Create PR with description
   - Address review comments
   - Merge after approval

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        files: ^frontend/.*\.(js|jsx|css|md)$
```

## 7. Continuous Integration/Continuous Deployment (CI/CD)

### ML Workflow

```yaml
name: ML Checks

on:
  push:
    paths:
      - "ML/**"

jobs:
  ml-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          cd ML
          pip install -r requirements.txt

      - name: Quality checks
        run: |
          black . --check
          isort . --check
          flake8

      - name: Run tests
        run: pytest tests/
```

### Backend Workflow

```yaml
name: Backend Checks

on:
  push:
    paths:
      - "backend/**"

jobs:
  backend-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt

      - name: Quality checks
        run: |
          black . --check
          isort . --check
          flake8

      - name: Run tests
        run: pytest tests/
```

### Frontend Workflow

```yaml
name: Frontend Checks

on:
  push:
    paths:
      - "frontend/**"

jobs:
  frontend-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Node
        uses: actions/setup-node@v2
        with:
          node-version: "16"

      - name: Install dependencies
        run: |
          cd frontend
          npm ci

      - name: Quality checks
        run: |
          npm run lint
          npm run format:check

      - name: Run tests
        run: npm test
```

### Combined Workflow

```yaml
name: Full Project Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      # Python Setup
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      # Node Setup
      - name: Set up Node
        uses: actions/setup-node@v2
        with:
          node-version: "16"

      # Run all checks
      - name: Run all checks
        run: |
          # ML
          cd ML && pip install -r requirements.txt && pytest tests/

          # Backend
          cd ../backend && pip install -r requirements.txt && pytest tests/

          # Frontend
          cd ../frontend && npm ci && npm test
```

## 8. Quality Assurance

### Required Checks

- All tests passing
- Code formatting verified
- Linting rules satisfied
- No merge conflicts
- PR description complete

### Branch Protection

Configure in GitHub:

1. Require status checks
2. Require up-to-date branches
3. Require review approvals
4. No direct pushes to main

## 9. Getting Started

1. **Clone Repository**

   ```bash
   git clone [repository-url]
   ```

2. **Install Pre-commit Hooks**

   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. **Set Up Components**

   ```bash
   # ML Setup
   cd ML
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Backend Setup
   cd ../backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Frontend Setup
   cd ../frontend
   npm install
   ```

4. **Verify Setup**
   ```bash
   # Run all tests
   cd ML && pytest
   cd ../backend && pytest
   cd ../frontend && npm test
   ```

## 10. Best Practices

1. **Code Organization**

   - Keep components focused
   - Document significant changes
   - Follow consistent patterns
   - Handle errors appropriately

2. **Testing**

   - Write tests for new features
   - Test error cases
   - Keep tests focused
   - Mock external dependencies

3. **Documentation**

   - Update README files
   - Document API changes
   - Keep this guide updated
   - Comment complex logic

4. **Communication**
   - Use clear PR descriptions
   - Document decisions
   - Share significant changes
   - Ask for help when needed

## Conclusion

This development guide is a living document. As our project evolves, we'll update it to reflect new best practices and lessons learned. Remember to:

- Keep code quality high
- Run tests regularly
- Document changes
- Communicate with the team
