# LLM Safety: ShieldGemma

This example demonstrates AI safety implementation using BentoML to filter potentially harmful language model inputs before processing. It combines ShieldGemma for safety evaluation with GPT-4o for response generation.

## Overview

ShieldGemma provides:
- **Proactive Safety Filtering**: Check inputs before processing
- **Customizable Thresholds**: Adjust safety sensitivity
- **Integration Ready**: Works with any LLM backend
- **Production Safe**: Enterprise-grade safety measures

## Architecture

The application uses two BentoML Services:
1. **Gemma Service**: Safety evaluation using ShieldGemma
2. **ShieldAssistant Service**: Orchestrates safety checks and response generation

## Implementation

### Safety Check Service
```python
@bentoml.service(
    resources={"gpu": 1, "gpu_type": "nvidia-l4"},
    traffic={"timeout": 120}
)
class Gemma:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/shieldgemma-2b")
    
    @bentoml.api
    async def check(self, prompt: str) -> ShieldResponse:
        """Evaluate safety of input prompt"""
        # Safety evaluation logic
        safety_score = self.evaluate_safety(prompt)
        
        return ShieldResponse(
            is_safe=safety_score < 0.5,
            safety_score=safety_score,
            prompt=prompt
        )
```

### Assistant with Safety
```python
@bentoml.service(
    resources={"cpu": "1000m"},
    envs=[{"name": "OPENAI_API_KEY"}]
)
class ShieldAssistant:
    shield = bentoml.depends(Gemma)
    
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
    
    @bentoml.api
    async def generate(
        self, 
        prompt: str, 
        threshold: float = 0.6
    ) -> AssistantResponse:
        """Generate response with safety check"""
        
        # Check prompt safety
        shield_result = await self.shield.check(prompt)
        
        if shield_result.safety_score > threshold:
            raise ValueError(f"Prompt exceeds safety threshold: {shield_result.safety_score}")
        
        # Generate safe response
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return AssistantResponse(
            content=response.choices[0].message.content,
            safety_score=shield_result.safety_score
        )
```

## Data Models
```python
from pydantic import BaseModel

class ShieldResponse(BaseModel):
    is_safe: bool
    safety_score: float
    prompt: str

class AssistantResponse(BaseModel):
    content: str
    safety_score: float
```

## Deployment

### BentoCloud
```bash
pip install bentoml
bentoml cloud login
bentoml secret create openai OPENAI_API_KEY=<your_key>
bentoml secret create huggingface HF_TOKEN=<your_token>
bentoml deploy .
```

### Local Serving
```bash
git clone https://github.com/bentoml/BentoShield.git
pip install -r requirements.txt
export OPENAI_API_KEY=<your_key>
bentoml serve
```

## Usage Examples

### Python Client
```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    try:
        response = client.generate(
            prompt="How do I bake a cake?",
            threshold=0.6
        )
        print(response.content)
    except ValueError as e:
        print(f"Safety filter triggered: {e}")
```

### Custom Thresholds
```python
# Strict safety (low threshold)
response = client.generate(prompt, threshold=0.3)

# Relaxed safety (high threshold) 
response = client.generate(prompt, threshold=0.8)
```

## Production Features

### Monitoring
- Safety score tracking
- Blocked request analytics
- Performance metrics

### Configuration
- Adjustable safety thresholds
- Custom safety models
- Allowlist/blocklist support

### Integration
- Works with any LLM backend
- Batch processing support
- Real-time filtering