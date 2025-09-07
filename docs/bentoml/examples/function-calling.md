# Function Calling Agent

This example demonstrates building an AI agent capable of currency conversion using function calling with Llama 3.1 70B. The agent can process natural language queries and execute specific functions to provide accurate results.

## Overview

Function calling enables LLMs to:
- Understand when to use external tools
- Extract parameters from natural language
- Execute functions with proper arguments
- Generate natural language responses from results

## Architecture

The system consists of two main BentoML Services:

1. **LLM Service (Llama)**: Processes queries and manages function calling
2. **Exchange Assistant Service**: Orchestrates the overall workflow

### Workflow
1. User submits query (e.g., "Convert 42 USD to CAD")
2. Exchange Assistant processes the query
3. LLM determines appropriate function and parameters
4. Exchange function computes currency conversion
5. LLM generates natural language response

## Code Implementation

### LLM Service Setup
```python
MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"

@bentoml.service(
    traffic={"timeout": 300},
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class Llama:
    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)
    
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model.path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model.path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
```

### Function Definition
```python
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get real-time exchange rate between two currencies"""
    # Implementation using exchange rate API
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url)
    data = response.json()
    return data["rates"][to_currency]

def convert_currency(from_currency: str, to_currency: str, amount: float) -> float:
    """Convert currency using real-time exchange rates"""
    rate = get_exchange_rate(from_currency, to_currency)
    return amount * rate
```

### Function Calling API
```python
@bentoml.api
def exchange(self, query: str) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert from one currency to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_currency": {
                            "type": "string",
                            "description": "Source currency code (e.g., USD, EUR)"
                        },
                        "to_currency": {
                            "type": "string", 
                            "description": "Target currency code (e.g., CAD, GBP)"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Amount to convert"
                        },
                    },
                    "required": ["from_currency", "to_currency", "amount"],
                },
            },
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]
    
    # Generate function call
    inputs = self.tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(self.model.device)
    
    with torch.no_grad():
        outputs = self.model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    # Parse and execute function call
    if "convert_currency" in response:
        # Extract parameters and call function
        result = self.execute_function_call(response)
        
        # Generate final response
        final_messages = messages + [
            {"role": "assistant", "content": response},
            {"role": "tool", "content": str(result)},
        ]
        
        return self.generate_final_response(final_messages)
    
    return response
```

### Assistant Service
```python
@bentoml.service(
    resources={"cpu": "1000m"},
    traffic={"concurrency": 10}
)
class ExchangeAssistant:
    llm = bentoml.depends(Llama)
    
    @bentoml.api
    def chat(self, query: str = "Convert 100 USD to EUR") -> str:
        """Process currency exchange queries using function calling"""
        return self.llm.exchange(query)
```

## Deployment Options

### BentoCloud Deployment

1. **Install and Setup**
   ```bash
   pip install bentoml
   bentoml cloud login
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/bentoml/BentoFunctionCalling.git
   cd BentoFunctionCalling
   ```

3. **Create Secrets**
   ```bash
   bentoml secret create huggingface HF_TOKEN=<your_token>
   bentoml secret create exchange-api EXCHANGE_API_KEY=<your_key>
   ```

4. **Deploy**
   ```bash
   bentoml deploy .
   ```

### Local Serving

1. **Requirements**
   - NVIDIA GPU with sufficient VRAM (A100 recommended)
   - CUDA toolkit installed
   - Python 3.8+

2. **Setup**
   ```bash
   git clone https://github.com/bentoml/BentoFunctionCalling.git
   cd BentoFunctionCalling
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   ```bash
   export HF_TOKEN=<your_huggingface_token>
   export EXCHANGE_API_KEY=<your_exchange_api_key>
   ```

4. **Serve**
   ```bash
   bentoml serve service:ExchangeAssistant
   ```

## Usage Examples

### Python Client
```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    # Simple conversion
    result = client.chat("Convert 100 USD to EUR")
    print(result)
    
    # Complex query
    result = client.chat("I have 500 British pounds, how much is that in Japanese yen?")
    print(result)
    
    # Multiple currencies
    result = client.chat("What's 1000 CAD in USD, EUR, and GBP?")
    print(result)
```

### HTTP API
```bash
curl -X POST "http://localhost:3000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Convert 250 EUR to USD"
     }'
```

### Example Conversations

**Input**: "I want to exchange 42 US dollars to Canadian dollars"

**Output**: "I can help you convert 42 US dollars to Canadian dollars. Based on the current exchange rate of 1.35, 42 USD equals approximately 56.70 CAD."

**Input**: "How much is 1000 euros in British pounds?"

**Output**: "1000 euros is equivalent to approximately 860 British pounds at the current exchange rate of 0.86."

## Advanced Features

### Multi-Function Support
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert between currencies",
            # ... parameters
        },
    },
    {
        "type": "function", 
        "function": {
            "name": "get_historical_rate",
            "description": "Get historical exchange rate",
            # ... parameters
        },
    },
]
```

### Error Handling
```python
def execute_function_call(self, function_call_str: str) -> str:
    try:
        # Parse function call
        func_data = json.loads(function_call_str)
        
        # Validate parameters
        if not all(key in func_data for key in ["from_currency", "to_currency", "amount"]):
            return "Error: Missing required parameters"
            
        # Execute function
        result = convert_currency(**func_data)
        return f"Conversion result: {result}"
        
    except Exception as e:
        return f"Error executing function: {str(e)}"
```

### Response Formatting
```python
def format_currency_response(self, amount: float, from_curr: str, to_curr: str, result: float) -> str:
    """Format the currency conversion response"""
    return f"{amount} {from_curr} equals {result:.2f} {to_curr} at the current exchange rate."
```

## Production Considerations

### Performance
- Use appropriate GPU instances (A100, H100)
- Implement request batching for multiple queries
- Cache exchange rates to reduce API calls
- Optimize model loading and memory usage

### Reliability
- Handle API rate limits gracefully
- Implement fallback mechanisms for exchange rate APIs
- Add comprehensive error handling
- Monitor function execution success rates

### Security
- Validate all function parameters
- Sanitize user inputs
- Secure API key management
- Implement rate limiting

### Monitoring
- Track function call success rates
- Monitor exchange rate API usage
- Log query patterns and performance metrics
- Set up alerts for failures

## Extending the Example

### Additional Functions
- Historical exchange rates
- Currency trend analysis
- Multi-currency calculations
- Investment calculations

### Integration Options
- Slack/Discord bots
- Web applications
- Mobile apps
- Voice assistants