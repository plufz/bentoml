# BentoML Examples Overview

BentoML provides a comprehensive collection of example projects across various AI and machine learning domains. These examples demonstrate best practices for deploying different types of models and building compound AI systems.

## Large Language Models (LLMs)

### Deployed Models
- **DeepSeek R1 Distill of Llama 3.3 70B** - Latest reasoning model
- **Llama 4 Scout** - Next-generation Llama model
- **Mistral Small 24B** - Efficient multilingual model

### Inference Runtimes
BentoML supports multiple high-performance inference engines:
- **vLLM** - High-throughput LLM serving
- **TensorRT-LLM** - NVIDIA optimized inference
- **LMDeploy** - MMRazor deployment framework
- **MLC-LLM** - Machine learning compilation
- **SGLang** - Structured generation language
- **Hugging Face TGI** - Text Generation Inference
- **Triton Inference Server** - NVIDIA's inference serving

## Compound AI Systems

Build sophisticated AI applications that combine multiple models and tools:

- **Function calling agent** - LLMs with external function capabilities
- **LangGraph agent** - Stateful multi-actor applications
- **CrewAI multi-agent system** - Collaborative AI agents
- **LLM safety with ShieldGemma** - Content filtering and safety
- **RAG with LlamaIndex** - Retrieval-augmented generation
- **Voice assistants with Pipecat** - Real-time voice AI
- **Phone agent with Twilio** - Telephony integration
- **Multi-LLM routing** - Intelligent model selection

## Image and Video Generation

### Image Generation Models
- **ComfyUI workflow APIs** - Advanced diffusion workflows
- **Stable Diffusion 3.5 Large Turbo** - Latest high-quality generation
- **Stable Diffusion 3 Medium** - Balanced performance and quality
- **Stable Diffusion XL Turbo** - Fast single-step generation
- **ControlNet** - Precision image control

### Video Generation
- Advanced video synthesis models
- Real-time video processing pipelines

## Audio Processing

### Text-to-Speech
- **ChatTTS** - Conversational speech synthesis
- **XTTS** - Extended text-to-speech with streaming
- **Bark** - Realistic speech and sound effects

### Speech-to-Text
- **WhisperX** - Advanced speech recognition

### Audio Generation
- **Moshi** - Audio generation model

## Computer Vision

### Object Detection
- **YOLO** - Real-time object detection
- Various YOLO versions (v5, v8, v10, v11)

### Image Classification
- **ResNet** - Residual network architecture
- Custom classification models

### Optical Character Recognition
- **EasyOCR** - Multi-language text recognition
- Document processing pipelines

## Embeddings and Search

### Text Embeddings
- **SentenceTransformers** - Semantic text similarity
- **CLIP** - Vision-language understanding
- **ColPali** - Document understanding

### Search Applications
- Semantic search implementations
- Vector database integration

## Custom Model Frameworks

### MLOps Integration
- **MLflow** - Experiment tracking and model registry
- **XGBoost** - Gradient boosting framework

### Custom Implementations
- PyTorch custom models
- TensorFlow custom models
- Scikit-learn pipelines

## Specialized Applications

### Visual Question Answering
- **BLIP** - Image captioning and VQA
- Multi-modal understanding

### Time Series Forecasting
- **Moirai** - Universal time series model
- **Facebook Prophet** - Business forecasting

### Document Processing
- PDF analysis and processing
- Document classification

## Getting Started

Each example includes:
- **Complete source code** - Ready-to-run implementations
- **Deployment configurations** - BentoCloud and local serving
- **Client examples** - Python and HTTP clients
- **Performance optimizations** - Best practices for production

### Repository Links
All examples are available as open-source GitHub repositories:
- Detailed README files
- Requirements and dependencies
- Step-by-step deployment guides
- Performance benchmarks

## Example Categories

### By Complexity
- **Beginner** - Simple model serving examples
- **Intermediate** - Multi-model applications
- **Advanced** - Complex AI systems with multiple components

### By Deployment Target
- **Local Development** - Quick prototyping and testing
- **Production Serving** - Scalable cloud deployment
- **Edge Deployment** - Resource-constrained environments

### By Use Case
- **Real-time Inference** - Low-latency applications
- **Batch Processing** - High-throughput workloads
- **Interactive Applications** - User-facing AI services

## Best Practices

The examples demonstrate:
- **Resource Management** - Efficient GPU/CPU utilization
- **Scaling Strategies** - Auto-scaling and load balancing
- **Monitoring** - Observability and debugging
- **Security** - Authentication and data protection
- **Performance** - Optimization techniques