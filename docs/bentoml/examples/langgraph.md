# LangGraph Agent: Stateful Multi-Actor Applications

This example demonstrates building a stateful, multi-actor agent application using LangGraph and BentoML. The agent can retrieve real-time information using search tools and process complex queries using LLMs.

## Overview

LangGraph enables creation of:
- **Stateful Applications**: Maintain conversation context across requests
- **Multi-Actor Systems**: Different components handle specialized tasks
- **Tool Integration**: External APIs and functions seamlessly integrated
- **Complex Workflows**: Multi-step reasoning and decision making

## Architecture

The application consists of two main services:
1. **LLM Service**: Handles language model inference
2. **SearchAgentService**: Orchestrates the agent workflow with tools

### Workflow Components
- **Agent Node**: Processes queries and decides on actions
- **Tools Node**: Executes external information retrieval
- **State Management**: Maintains conversation history and context

## Code Implementation

### LLM Service Configuration
```python
ENGINE_CONFIG = {
    "model": "mistralai/Ministral-8B-Instruct-2410",
    "tokenizer_mode": "mistral", 
    "max_model_len": 4096,
    "enable_prefix_caching": False,
}

@bentoml.service(
    traffic={"timeout": 300},
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
    envs=[
        {"name": "HF_TOKEN"},
    ],
)
class LLM:
    model_id = ENGINE_CONFIG["model"]
    
    def __init__(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**ENGINE_CONFIG)
        )
    
    @bentoml.api
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        from vllm import SamplingParams
        import uuid
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        request_id = str(uuid.uuid4())
        results = self.engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in results:
            return request_output.outputs[0].text
```

### Search Agent Service
```python
@bentoml.service(
    workers=2,
    resources={"cpu": "2000m"},
    traffic={
        "concurrency": 16,
        "external_queue": True,
    }
)
class SearchAgentService:
    llm = bentoml.depends(LLM)
    
    def __init__(self):
        from langgraph.graph import StateGraph, END
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Initialize search tool
        self.search_tool = DuckDuckGoSearchRun()
        
        # Create the agent workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tools_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        self.app = workflow.compile()
```

### Agent State Management
```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_queries: List[str]
    search_results: List[str]
```

### Agent and Tools Nodes
```python
async def agent_node(self, state: AgentState):
    """Main agent reasoning node"""
    messages = state["messages"]
    
    # Create system prompt
    system_prompt = """You are a helpful AI assistant that can search for information.
    When you need current information, use the search tool.
    Always provide helpful and accurate responses."""
    
    # Prepare prompt for LLM
    conversation = f"{system_prompt}\n\n"
    for msg in messages:
        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"
    
    # Generate response
    response = await self.llm.generate(
        prompt=conversation,
        max_tokens=512,
        temperature=0.3,
    )
    
    # Determine if search is needed
    if self.needs_search(response):
        return {
            "messages": [AIMessage(content="I need to search for more information.")],
            "search_queries": [self.extract_search_query(messages[-1].content)],
        }
    
    return {"messages": [AIMessage(content=response)]}

async def tools_node(self, state: AgentState):
    """Execute search tools"""
    search_queries = state.get("search_queries", [])
    search_results = []
    
    for query in search_queries:
        try:
            result = self.search_tool.run(query)
            search_results.append(result)
        except Exception as e:
            search_results.append(f"Search failed: {str(e)}")
    
    # Generate response based on search results
    if search_results:
        context = "\n".join(search_results)
        prompt = f"Based on this information:\n{context}\n\nAnswer the user's question: {state['messages'][-1].content}"
        
        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3,
        )
        
        return {
            "messages": [AIMessage(content=response)],
            "search_results": search_results,
        }
    
    return {"messages": [AIMessage(content="No search results found.")]}
```

### Async Task API
```python
@bentoml.task
async def invoke(
    self,
    input_query: str = "What is the weather in San Francisco today?",
) -> str:
    """Process user query through the agent workflow"""
    
    # Initialize state with user message
    initial_state = {
        "messages": [HumanMessage(content=input_query)],
        "search_queries": [],
        "search_results": [],
    }
    
    # Run the workflow
    final_state = await self.app.ainvoke(initial_state)
    
    # Return the final response
    return final_state["messages"][-1].content

@bentoml.api
async def chat(
    self,
    message: str,
    session_id: str = "default",
) -> str:
    """Chat interface with session management"""
    return await self.invoke(message)
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
   git clone https://github.com/bentoml/BentoLangGraph.git
   cd BentoLangGraph
   ```

3. **Create Secrets**
   ```bash
   bentoml secret create huggingface HF_TOKEN=<your_token>
   bentoml secret create anthropic ANTHROPIC_API_KEY=<your_key>  # If using Claude
   ```

4. **Deploy**
   ```bash
   bentoml deploy .
   ```

### Local Serving

1. **Setup Environment**
   ```bash
   git clone https://github.com/bentoml/BentoLangGraph.git
   cd BentoLangGraph
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   ```bash
   export HF_TOKEN=<your_huggingface_token>
   export ANTHROPIC_API_KEY=<your_anthropic_key>  # Optional
   ```

3. **Serve Locally**
   ```bash
   bentoml serve service:SearchAgentService
   ```

## Usage Examples

### Python Client
```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    # Real-time information query
    result = client.chat("What's the latest news about AI developments?")
    print(result)
    
    # Weather query
    result = client.chat("What's the weather like in Tokyo today?")
    print(result)
    
    # Complex research question
    result = client.chat("Compare the latest features of GPT-4 and Claude 3.5")
    print(result)
```

### Async Task Usage
```python
import asyncio
import bentoml

async def main():
    async with bentoml.AsyncHTTPClient("http://localhost:3000") as client:
        # Submit async task
        task = await client.invoke.submit("Research quantum computing breakthroughs in 2024")
        
        # Get result when ready
        result = await task.get()
        print(result)

asyncio.run(main())
```

### HTTP API
```bash
# Chat endpoint
curl -X POST "http://localhost:3000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What are the top tech trends for 2024?",
       "session_id": "user123"
     }'

# Async task endpoint
curl -X POST "http://localhost:3000/invoke" \
     -H "Content-Type: application/json" \
     -d '{
       "input_query": "Latest developments in renewable energy"
     }'
```

## Advanced Features

### Multi-Model Support
```python
@bentoml.service
class MultiModelAgent:
    mistral_llm = bentoml.depends(MistralLLM)
    claude_llm = bentoml.depends(ClaudeLLM)
    
    async def route_to_best_model(self, query: str, task_type: str):
        """Route queries to the most appropriate model"""
        if task_type == "reasoning":
            return await self.claude_llm.generate(query)
        elif task_type == "code":
            return await self.mistral_llm.generate(query)
        else:
            return await self.mistral_llm.generate(query)
```

### Custom Tools Integration
```python
def create_custom_tools(self):
    """Create domain-specific tools"""
    from langchain.tools import BaseTool
    
    class WeatherTool(BaseTool):
        name = "weather_lookup"
        description = "Get current weather information"
        
        def _run(self, location: str) -> str:
            # Implementation using weather API
            return f"Weather in {location}: Sunny, 72°F"
    
    class StockTool(BaseTool):
        name = "stock_price"
        description = "Get current stock price"
        
        def _run(self, symbol: str) -> str:
            # Implementation using stock API
            return f"Current price of {symbol}: $150.25"
    
    return [WeatherTool(), StockTool()]
```

### Session Management
```python
class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session_state(self, session_id: str) -> AgentState:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "search_queries": [],
                "search_results": [],
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, new_state: AgentState):
        self.sessions[session_id] = new_state
```

## Production Considerations

### Performance Optimization
- **Model Caching**: Cache model responses for common queries
- **Concurrent Processing**: Handle multiple agent workflows simultaneously
- **Search Rate Limiting**: Manage external API call rates
- **Memory Management**: Clean up old sessions periodically

### Error Handling
```python
async def robust_agent_node(self, state: AgentState):
    try:
        return await self.agent_node(state)
    except Exception as e:
        logger.error(f"Agent node error: {e}")
        return {
            "messages": [AIMessage(content="I encountered an error processing your request. Please try again.")]
        }
```

### Monitoring and Logging
- Track agent decision paths
- Monitor search tool usage
- Log conversation flows
- Performance metrics collection

### Scaling Strategies
- Horizontal scaling with multiple replicas
- Load balancing across agent instances
- Distributed state management
- Caching layers for common queries