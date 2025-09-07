# Configuration Directory

This directory contains all configuration files for the BentoML project, organized for better maintainability.

## Structure

```
config/
├── README.md              # This file
├── bentoml.yaml           # Main BentoML server configuration
└── bentofiles/            # Service-specific Bento build configurations
    ├── default.yaml       # Default bentofile for basic services
    ├── stable-diffusion.yaml  # Configuration for Stable Diffusion service
    ├── llava.yaml         # Configuration for LLaVA vision service
    └── whisper.yaml       # Configuration for Whisper transcription service
```

## Usage

### BentoML Server Configuration
The `bentoml.yaml` file contains server-level settings used by the `run_bentoml.sh` script.

### Service Build Configurations
The `bentofiles/` directory contains service-specific build configurations:

- **default.yaml**: Basic configuration for simple services
- **stable-diffusion.yaml**: Specialized config for image generation with GPU requirements
- **llava.yaml**: Configuration for vision-language model with custom dependencies
- **whisper.yaml**: Configuration for audio transcription with MLX dependencies

### Building with Custom Configurations
Use the `BENTOFILE` environment variable to specify a custom configuration:

```bash
# Build with specific service configuration
BENTOFILE=config/bentofiles/stable-diffusion.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

# Build with default configuration (no BENTOFILE needed)
./scripts/run_bentoml.sh build services/example_service.py
```

## Benefits

1. **Organization**: All configuration files in one place
2. **Clarity**: Clear naming conventions for different services
3. **Maintainability**: Easy to manage and update configurations
4. **Scalability**: Simple to add new service configurations
5. **Clean Root**: Keeps the project root directory uncluttered