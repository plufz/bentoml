# vLLM: High-Performance LLM Serving

This example demonstrates how to serve large language models efficiently using vLLM with BentoML, providing OpenAI-compatible APIs for seamless integration.

## Overview

vLLM is a high-performance inference engine designed for large language models. This example shows how to:
- Serve Llama 3.1 8B Instruct model with vLLM
- Provide OpenAI-compatible chat endpoints
- Deploy on BentoCloud or serve locally
- Optimize performance with advanced configurations

## Key Features

- **High Throughput**: Optimized for serving multiple concurrent requests
- **OpenAI Compatibility**: Standard chat completion API
- **Flexible Configuration**: Customizable engine parameters
- **Production Ready**: Built-in monitoring and scaling

## Configuration

### Engine Configuration
```python
ENGINE_CONFIG = {
    'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'max_model_len': 2048,
    'dtype': 'half',
    'enable_prefix_caching': True,
}
```

### Service Configuration
```python
@bentoml.service(
    name='bentovllm-llama3.1-8b-instruct-service',
    traffic={'timeout': 300},
    resources={'gpu': 1, 'gpu_type': 'nvidia-l4'},
    envs=[{'name': 'HF_TOKEN'}],
)
class VLLM:
    def __init__(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**ENGINE_CONFIG)
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
```

## API Endpoints

### Generate Endpoint
```python
@bentoml.api
async def generate(
    self,
    prompt: str = "Explain the importance of fast language model serving",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    stream = await self.engine.add_request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
    
    async for request_output in stream:
        yield request_output.outputs[0].text
```

### Chat Completion (OpenAI Compatible)
```python
@bentoml.api
async def chat_completions(
    self,
    messages: List[ChatMessage],
    model: str = ENGINE_CONFIG["model"],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> ChatCompletionResponse:
    # OpenAI-compatible chat completion implementation
    # ...
```

## Deployment Options

### BentoCloud Deployment

1. **Install BentoML**
   ```bash
   pip install bentoml
   ```

2. **Log in to BentoCloud**
   ```bash
   bentoml cloud login
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/bentoml/BentoVLLM.git
   cd BentoVLLM
   ```

4. **Create Hugging Face Secret**
   ```bash
   bentoml secret create huggingface HF_TOKEN=<your_hf_token>
   ```

5. **Deploy**
   ```bash
   bentoml deploy .
   ```

### Local Serving

1. **Clone Repository**
   ```bash
   git clone https://github.com/bentoml/BentoVLLM.git
   cd BentoVLLM
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Hugging Face Token**
   ```bash
   export HF_TOKEN=<your_hf_token>
   ```

4. **Serve Locally**
   ```bash
   bentoml serve service:VLLM
   ```

## Client Examples

### Python Client (BentoML)
```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Who are you? Please respond in pirate speak!",
        max_tokens=1024,
        temperature=0.8,
    )
    
    for response in response_generator:
        print(response, end='', flush=True)
```

### OpenAI-Compatible Client
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:3000/v1',
    api_key='na'  # Not required for local serving
)

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user", 
            "content": "Explain quantum computing in simple terms"
        }
    ],
    max_tokens=500,
    temperature=0.7,
)

print(chat_completion.choices[0].message.content)
```

### Streaming Chat
```python
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=1000,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL Example
```bash
curl -X POST "http://localhost:3000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is the future of AI?",
       "max_tokens": 500,
       "temperature": 0.7
     }'
```

## Performance Optimization

### Engine Parameters
- **max_model_len**: Maximum sequence length
- **dtype**: Data type (half, float, auto)
- **enable_prefix_caching**: Cache common prefixes
- **gpu_memory_utilization**: GPU memory usage fraction

### Advanced Configuration
```python
ENGINE_CONFIG = {
    'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'max_model_len': 4096,
    'dtype': 'auto',
    'enable_prefix_caching': True,
    'gpu_memory_utilization': 0.95,
    'max_num_batched_tokens': 8192,
    'max_num_seqs': 256,
}
```

### Resource Planning
- **Single GPU**: L4, A10G for 8B models
- **Multi-GPU**: A100, H100 for larger models
- **Memory**: 16GB+ for 8B models, 40GB+ for 70B models

## Production Considerations

### Monitoring
- Request throughput and latency
- GPU utilization and memory usage
- Error rates and timeout handling

### Scaling
- Horizontal scaling with multiple replicas
- Auto-scaling based on request queue
- Load balancing across instances

### Security
- API key authentication
- Rate limiting
- Input validation and sanitization

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce max_model_len or gpu_memory_utilization
2. **Slow Performance**: Enable prefix caching and optimize batch size
3. **Model Loading**: Verify HF_TOKEN and model access permissions

### Performance Tips
- Use appropriate GPU types for your model size
- Enable tensor parallelism for large models
- Monitor and adjust batching parameters
- Use FP16 or BF16 for memory efficiency