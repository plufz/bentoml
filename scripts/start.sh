#!/bin/bash

# Start BentoML services - either multi-service (default) or individual service by shortname
# Usage: ./scripts/start.sh [service_name]
# Examples:
#   ./scripts/start.sh                  # Start multi-service (default)
#   ./scripts/start.sh llava            # Start LLaVA service
#   ./scripts/start.sh stable_diffusion # Start Stable Diffusion service
#   ./scripts/start.sh whisper          # Start Whisper service
#   ./scripts/start.sh example          # Start Example service

SERVICE_NAME=${1:-}

case $SERVICE_NAME in
    "llava")
        echo "Starting LLaVA service..."
        ./scripts/run_bentoml.sh serve services.llava_service:LLaVAService
        ;;
    "stable_diffusion")
        echo "Starting Stable Diffusion service..."
        ./scripts/run_bentoml.sh serve services.stable_diffusion_service:StableDiffusionService
        ;;
    "whisper")
        echo "Starting Whisper service..."
        ./scripts/run_bentoml.sh serve services.whisper_service:WhisperService
        ;;
    "example")
        echo "Starting Example service..."
        ./scripts/run_bentoml.sh serve services.example_service:ExampleService
        ;;
    "multi" | "")
        echo "Starting Multi-Service (default)..."
        ./scripts/run_bentoml.sh serve services.multi_service:MultiService
        ;;
    *)
        echo "Unknown service: $SERVICE_NAME"
        echo ""
        echo "Available services:"
        echo "  llava            - LLaVA vision-language service"
        echo "  stable_diffusion - Stable Diffusion image generation service"
        echo "  whisper          - Whisper audio transcription service"
        echo "  example          - Example hello service"
        echo "  multi            - Multi-service (all services combined)"
        echo ""
        echo "Usage: ./scripts/start.sh [service_name]"
        echo "Default: multi-service if no argument provided"
        exit 1
        ;;
esac