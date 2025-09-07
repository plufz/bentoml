#!/bin/bash

# Build all BentoML services
echo "Building Example Service..."
./scripts/run_bentoml.sh build services/example_service.py

echo "Building Stable Diffusion Service..."
BENTOFILE=config/bentofiles/stable-diffusion.yaml ./scripts/run_bentoml.sh build services/stable_diffusion_service.py

echo "Building LLaVA Service..."
BENTOFILE=config/bentofiles/llava.yaml ./scripts/run_bentoml.sh build services/llava_service.py

echo "Building Whisper Service..."
BENTOFILE=config/bentofiles/whisper.yaml ./scripts/run_bentoml.sh build services/whisper_service.py

echo "Building Photo Upscaler Service..."
BENTOFILE=config/bentofiles/upscaler.yaml ./scripts/run_bentoml.sh build services/upscaler_service.py

echo "Building RAG Service..."
BENTOFILE=config/bentofiles/rag.yaml ./scripts/run_bentoml.sh build services/rag_service.py

echo "Building Multi-Service (unified deployment)..."
BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh build services/multi_service.py

echo "All services built successfully!"