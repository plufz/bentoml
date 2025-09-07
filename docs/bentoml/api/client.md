# BentoML Client API

The BentoML client API provides both synchronous and asynchronous HTTP clients for interacting with BentoML services.

## SyncHTTPClient

### Class Definition
```python
class SyncHTTPClient(
    url: str, 
    *, 
    token: str | None = None, 
    timeout: float = 30, 
    server_ready_timeout: float | None = None
)
```

### Parameters
- `url`: URL of the BentoML service
- `token`: Authentication token (optional)
- `timeout`: Client timeout, defaults to 30 seconds
- `server_ready_timeout`: Timeout for server readiness (optional)

### Example Usage
```python
with SyncHTTPClient("http://localhost:3000") as client:
    resp = client.call("classify", input_series=[[1,2,3,4]])
    assert resp == [0]
    
    # Or using named method directly
    resp = client.classify(input_series=[[1,2,3,4]])
    assert resp == [0]
```

### Methods
- `client_cls`: Alias of `Client`
- `is_ready(timeout: int | None = None) -> bool`: Check if service is ready

## AsyncHTTPClient

### Class Definition
```python
class AsyncHTTPClient(
    url: str, 
    *, 
    token: str | None = None, 
    timeout: float = 30, 
    server_ready_timeout: float | None = None
)
```

### Parameters
- Same as SyncHTTPClient

### Example Usage
```python
async with AsyncHTTPClient("http://localhost:3000") as client:
    resp = await client.call("classify", input_series=[[1,2,3,4]])
    assert resp == [0]
    
    # Streaming example
    resp = client.stream(prompt="hello")
    async for data in resp:
        print(data)
```

### Methods
- `client_cls`: Alias of `AsyncClient`
- `is_ready(timeout: int | None = None) -> bool`: Async method to check service readiness

## Key Differences

- `SyncHTTPClient` uses synchronous method calls
- `AsyncHTTPClient` uses asynchronous method calls with `await`
- Both support direct method calls and generic `call()` method
- Async client supports streaming responses

## Usage Patterns

### Context Manager Pattern
Both clients should be used with context managers to ensure proper resource cleanup:

```python
# Sync
with SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(data)

# Async  
async with AsyncHTTPClient("http://localhost:3000") as client:
    result = await client.predict(data)
```

### Authentication
When working with authenticated services:

```python
client = SyncHTTPClient("http://localhost:3000", token="your-api-token")
```

### Timeout Configuration
Custom timeout settings for different scenarios:

```python
# Short timeout for health checks
quick_client = SyncHTTPClient("http://localhost:3000", timeout=5)

# Longer timeout for complex inference
inference_client = SyncHTTPClient("http://localhost:3000", timeout=120)
```