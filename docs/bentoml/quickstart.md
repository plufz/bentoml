# BentoML Quickstart Guide

Complete guide to getting started with BentoML from installation to deployment.

## Prerequisites

- **Python 3.11** (recommended)
- **Virtual environment** (suggested for isolation)

## Setup Steps

### 1. Clone Repository

```bash
git clone https://github.com/bentoml/quickstart.git
cd quickstart
```

### 2. Create Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv quickstart
source quickstart/bin/activate
```

**Windows:**
```bash
python -m venv quickstart
quickstart\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install bentoml torch transformers
```

## Create Your First Service

Create a file called `service.py`:

```python
import bentoml
from transformers import pipeline

@bentoml.service
class Summarization:
    def __init__(self):
        self.pipeline = pipeline('summarization')

    @bentoml.api
    def summarize(self, text: str) -> str:
        result = self.pipeline(text)
        return f"Hello world! Here's your summary: {result[0]['summary_text']}"
```

## Serve Locally

Start the service:

```bash
bentoml serve
```

This will start a local server at `http://localhost:3000`

## Interact with Your Service

### 1. CURL Request

```bash
curl -X POST http://localhost:3000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text to summarize here..."}'
```

### 2. Python Client

```python
import requests

response = requests.post(
    "http://localhost:3000/summarize",
    json={"text": "Your long text to summarize here..."}
)
print(response.json())
```

### 3. Swagger UI

Open `http://localhost:3000` in your browser to access the interactive API documentation.

## Key Features Demonstrated

- **Hugging Face Integration** - Uses transformers pipeline for NLP tasks
- **Simple Service Creation** - Minimal code required for deployment
- **Multiple Interaction Methods** - REST API, Python client, web UI
- **Automatic API Generation** - Swagger/OpenAPI documentation included

## Next Steps

After mastering the basics, explore:

- **Batch Requests** - Process multiple inputs efficiently
- **Load Custom Models** - Deploy your own trained models  
- **Create Docker Images** - Containerize your service
- **Cloud Deployment** - Scale to production environments
- **Advanced Features** - Custom endpoints, middleware, monitoring

## Architecture Overview

1. **Service Definition** - Python class with `@bentoml.service` decorator
2. **API Endpoints** - Methods with `@bentoml.api` decorator
3. **Model Loading** - Initialize models in `__init__` method
4. **Request Handling** - Automatic serialization/deserialization
5. **Response Generation** - Return Python objects, auto-converted to JSON

This quickstart demonstrates a text summarization service but the same patterns apply to any AI/ML model deployment.