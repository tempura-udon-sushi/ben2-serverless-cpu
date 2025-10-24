# GitHub Repository Setup Guide

This guide walks you through creating a new GitHub repository for the CPU serverless worker.

## 📋 Prerequisites

- GitHub account
- Git installed locally
- Docker Hub account (for hosting images)

## 🚀 Step-by-Step Setup

### 1. Create GitHub Repository

```bash
# On GitHub.com:
# 1. Click "New Repository"
# 2. Name: ben2-serverless-cpu
# 3. Description: "CPU-optimized ComfyUI worker for BEN2 background removal"
# 4. Public or Private (your choice)
# 5. Don't initialize with README (we have one)
# 6. Create repository
```

### 2. Initialize Local Repository

```bash
# From ben2-serverless-cpu directory
cd /path/to/Comfy_vanila/ben2-serverless-cpu

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: BEN2 CPU serverless worker"
```

### 3. Link to GitHub and Push

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ben2-serverless-cpu.git

# Rename branch to main (if needed)
git branch -M main

# Push
git push -u origin main
```

### 4. Create Docker Hub Repository

```bash
# On hub.docker.com:
# 1. Click "Create Repository"
# 2. Name: ben2-serverless-cpu
# 3. Visibility: Public
# 4. Create

# Login to Docker Hub locally
docker login

# Tag and push your image
docker tag ben2-serverless-cpu:latest YOUR_DOCKERHUB_USERNAME/ben2-serverless-cpu:latest
docker push YOUR_DOCKERHUB_USERNAME/ben2-serverless-cpu:latest
```

## 📁 Repository Structure

```
ben2-serverless-cpu/
├── Dockerfile              # CPU-optimized Docker image
├── .dockerignore          # Files to exclude from build
├── .gitignore             # Git exclusions
├── README.md              # Main documentation
├── GITHUB_SETUP.md        # This file
├── build.sh               # Linux/Mac build script
├── build.bat              # Windows build script
└── examples/              # (Optional) Usage examples
    └── test_client.py
```

## 🔄 Workflow for Updates

### Making Changes

```bash
# 1. Make your changes to files
# 2. Test locally
docker build -t ben2-serverless-cpu:test .

# 3. Commit changes
git add .
git commit -m "Description of changes"

# 4. Push to GitHub
git push origin main

# 5. Build and tag new version
docker build -t ben2-serverless-cpu:1.1 .
docker tag ben2-serverless-cpu:1.1 YOUR_USERNAME/ben2-serverless-cpu:1.1
docker tag ben2-serverless-cpu:1.1 YOUR_USERNAME/ben2-serverless-cpu:latest

# 6. Push to Docker Hub
docker push YOUR_USERNAME/ben2-serverless-cpu:1.1
docker push YOUR_USERNAME/ben2-serverless-cpu:latest
```

## 🏷️ Version Tags

Use semantic versioning:
- `1.0.0` - Initial release
- `1.0.1` - Bug fixes
- `1.1.0` - New features
- `2.0.0` - Breaking changes

```bash
# Tag in git
git tag v1.0.0
git push origin v1.0.0

# Tag in Docker
docker tag ben2-serverless-cpu:latest YOUR_USERNAME/ben2-serverless-cpu:1.0.0
docker push YOUR_USERNAME/ben2-serverless-cpu:1.0.0
```

## 📝 Recommended README Badges

Add to your README.md:

```markdown
![Docker Image Size](https://img.shields.io/docker/image-size/YOUR_USERNAME/ben2-serverless-cpu)
![Docker Pulls](https://img.shields.io/docker/pulls/YOUR_USERNAME/ben2-serverless-cpu)
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/ben2-serverless-cpu)
![License](https://img.shields.io/github/license/YOUR_USERNAME/ben2-serverless-cpu)
```

## 🔐 Important Notes

### Files to Keep Private
- API keys
- Credentials
- Test images with sensitive content

### Files to Include
- ✅ Dockerfile
- ✅ Build scripts
- ✅ Documentation
- ✅ .dockerignore
- ✅ .gitignore
- ✅ LICENSE (choose appropriate)

### Don't Include in Git
- ❌ Large model files (they're downloaded during build)
- ❌ ComfyUI source (reference in README instead)
- ❌ Test images
- ❌ Build artifacts

## 🆚 Separate from GPU Version

Keep CPU and GPU versions in separate repositories:

```
ben2-serverless-cpu/     # This repo (CPU)
ben2-serverless-gpu/     # Separate repo (GPU)
```

Or use branches:
```bash
main           # GPU version
cpu            # CPU version
```

## 📚 Optional: Add GitHub Actions

Create `.github/workflows/docker-build.yml` for automated builds:

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            yourusername/ben2-serverless-cpu:latest
            yourusername/ben2-serverless-cpu:${{ github.ref_name }}
```

## ✅ Checklist

Before making repository public:

- [ ] Remove any API keys or secrets
- [ ] Add LICENSE file
- [ ] Complete README.md
- [ ] Test Dockerfile builds successfully
- [ ] Add .gitignore and .dockerignore
- [ ] Test on clean clone
- [ ] Add repository description
- [ ] Add topics/tags on GitHub

## 🤝 Contributing

If you want others to contribute:

1. Add CONTRIBUTING.md
2. Set up issue templates
3. Add code of conduct
4. Document development setup

---

**Ready to publish?** Follow the steps above and your repository will be live!
