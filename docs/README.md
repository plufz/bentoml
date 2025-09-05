# BentoML Local Setup Documentation

This directory contains comprehensive documentation for the BentoML local setup with multiple AI services.

## 📚 Documentation Index

### Getting Started
- **[Quick Start Guide](quick-start.md)** - Get up and running in 5 minutes
- **[Installation & Setup](installation.md)** - Detailed installation instructions
- **[Configuration](configuration.md)** - Environment and service configuration

### Services
- **[Stable Diffusion Service](stable-diffusion.md)** - Text-to-image generation
- **[LLaVA Service](llava-service.md)** - Vision-language analysis with structured JSON output
- **[Example Service](example-service.md)** - Simple example for testing

### Development
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Testing Guide](testing.md)** - How to test services and endpoints
- **[Utilities Documentation](utilities.md)** - Reusable utility modules

### Operations
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Performance Tuning](performance.md)** - Optimization tips for Apple Silicon and CUDA

## 🚀 Quick Navigation

| What you want to do | Go to |
|---------------------|-------|
| Set up the project for the first time | [Installation Guide](installation.md) |
| Generate images from text | [Stable Diffusion Service](stable-diffusion.md) |
| Analyze images with AI | [LLaVA Service](llava-service.md) |
| Test your services | [Testing Guide](testing.md) |
| Fix issues | [Troubleshooting](troubleshooting.md) |

## 🏗️ Architecture

This setup uses:
- **UV** for fast Python package management
- **BentoML 1.4+** with modern API patterns
- **Apple Silicon (MPS)** optimization for M-series Macs
- **Custom HuggingFace cache** support for external drive storage
- **Modular utilities** for easy service extension

## 🎯 Available Services

- **Stable Diffusion**: Text → Image generation
- **LLaVA**: Image + Text → Structured JSON analysis
- **Example**: Simple hello world service for testing

All services support:
- Automatic device detection (MPS/CUDA/CPU)
- Custom model caching
- Comprehensive health checks
- Swagger UI documentation