# BentoML Service Structure Guide

This guide explains how BentoML services are structured in this project and provides comprehensive documentation for creating, updating, and maintaining services.

## Overview

BentoML services in this project follow a consistent architecture pattern that promotes maintainability, testability, and scalability. This documentation is organized into focused guides covering different aspects of service development.

## Architecture Overview

```
bentoml/
├── services/           # Main service implementations
├── utils/             # Shared utilities and helpers
├── config/            # Service configurations
│   └── bentofiles/    # Bento build configurations
├── tests/             # Test files for all services
├── scripts/           # Development and deployment scripts
└── docs/              # Documentation files
```

## Documentation Structure

### Core Guides

1. **[Service Creation Guide](service-creation.md)** - Step-by-step guide for creating new services
   - Service implementation patterns
   - File structure and organization
   - Code examples and templates

2. **[Configuration Guide](configuration.md)** - Service configuration and build setup
   - Bento configuration files
   - Resource management
   - Environment variables and settings

3. **[Testing Guide](testing.md)** - Comprehensive testing strategy
   - Unit testing patterns
   - Integration testing setup
   - Test organization and best practices

4. **[Scripts Integration Guide](scripts-integration.md)** - Development workflow integration
   - Build script updates
   - Testing script configuration
   - Deployment automation

5. **[Multi-Service Integration](multi-service.md)** - Integrating with the multi-service architecture
   - Multi-service composition patterns
   - Endpoint integration
   - Unified service deployment

### Quick Reference

- **[File Checklist](file-checklist.md)** - Complete checklist for new services
- **[Best Practices](best-practices.md)** - Service design and development guidelines
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Getting Started

If you're new to developing services in this project:

1. Start with the **[Service Creation Guide](service-creation.md)** for a complete walkthrough
2. Review the **[Configuration Guide](configuration.md)** to understand build setup
3. Follow the **[Testing Guide](testing.md)** to implement proper testing
4. Use the **[File Checklist](file-checklist.md)** to ensure you've updated all necessary files

## Service Examples

Learn from existing service implementations:

- `services/example_service.py` - Basic service template
- `services/stable_diffusion_service.py` - Model-heavy service with GPU requirements
- `services/llava_service.py` - Vision-language service with image processing
- `services/whisper_service.py` - Audio processing service
- `services/upscaler_service.py` - Image enhancement service
- `services/rag_service.py` - RAG service with vector database integration
- `services/multi_service.py` - Multi-service composition pattern

## Development Workflow Integration

This project uses a comprehensive development workflow:

1. **Development**: Use `./scripts/start.sh` for local development
2. **Testing**: Use `./scripts/test.sh --service your_service`
3. **Health Check**: Use `./scripts/health.sh` to verify service status
4. **Endpoint Testing**: Use `./scripts/endpoint.sh your_endpoint '{}'`
5. **Building**: Use `./scripts/build_services.sh` for production builds

## Key Principles

- **Consistency**: All services follow the same structural patterns
- **Testability**: Comprehensive unit and integration testing
- **Modularity**: Clear separation between services, utilities, and configuration
- **Documentation**: Thorough documentation for all services and endpoints
- **Automation**: Integrated build, test, and deployment workflows

## Support and Maintenance

When working with services:
- Always update tests when modifying service logic
- Keep documentation current with code changes
- Follow the established patterns from existing services
- Update the multi-service integration when adding new endpoints
- Maintain consistent API patterns across all services

For specific guidance on any aspect of service development, refer to the detailed guides linked above.