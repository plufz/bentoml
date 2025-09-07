# BentoML Exceptions

BentoML provides a comprehensive set of custom exceptions to handle various error scenarios during service development and deployment. All exceptions inherit from the base `BentoMLException` class.

## Base Exception

### `BentoMLException`
The base class for all BentoML errors. All custom exceptions are derived from this class.

```python
class BentoMLException(Exception):
    """Base exception class for BentoML"""
    pass
```

**Usage:**
- Base class for custom BentoML exceptions
- Includes optional error code parameter for structured error handling
- Can be caught to handle any BentoML-related error

## Client and Remote Exceptions

### `RemoteException`
Wraps exceptions that occur on remote servers during client-server communication.

```python
class RemoteException(BentoMLException):
    """Exception for remote server errors"""
    pass
```

**When raised:**
- Remote BentoML service returns an error
- Network communication failures with remote services
- Authentication or authorization failures

**Attributes:**
- Includes payload for additional error context
- Contains original remote error information

## Input and Validation Exceptions

### `InvalidArgument`
Raised when unexpected or invalid arguments are received.

```python
class InvalidArgument(BentoMLException):
    """Invalid argument provided"""
    pass
```

**Common scenarios:**
- Invalid CLI arguments
- Malformed HTTP request parameters
- Incorrect API function parameters
- Invalid configuration values

**Examples:**
```python
# Invalid model name format
raise InvalidArgument("Model name must follow the format 'name:version'")

# Invalid parameter type
raise InvalidArgument(f"Expected int, got {type(value)}")
```

### `BadInput`
Raised when the API server receives an invalid input request.

```python
class BadInput(BentoMLException):
    """Bad input data provided to API"""
    pass
```

**Common scenarios:**
- Malformed JSON in request body
- Missing required input fields
- Input data type mismatch
- Invalid file uploads

**Examples:**
```python
# Invalid JSON format
raise BadInput("Request body must be valid JSON")

# Missing required field
raise BadInput("Missing required field 'image' in request")
```

## Dependency and Import Exceptions

### `MissingDependencyException`
Triggered when required dependencies fail to load.

```python
class MissingDependencyException(BentoMLException):
    """Required dependency is missing"""
    pass
```

**Common scenarios:**
- Missing optional dependencies (e.g., `pydantic` for JSON IODescriptor)
- Framework-specific libraries not installed (e.g., `torch`, `tensorflow`)
- Version incompatibility issues

**Attributes:**
- Supports optional component extensions
- Provides installation suggestions

**Examples:**
```python
# Missing optional dependency
raise MissingDependencyException(
    "pydantic is required for JSON validation. Install with: pip install pydantic"
)
```

### `ImportServiceError`
Occurs when BentoML fails to import a user's service file.

```python
class ImportServiceError(BentoMLException):
    """Failed to import service"""
    pass
```

**Common scenarios:**
- Python syntax errors in service file
- Missing dependencies in service code
- Circular import issues
- Invalid service definition

## Server and Runtime Exceptions

### `InternalServerError`
Raised for internal processing issues when valid arguments are received but internal problems occur.

```python
class InternalServerError(BentoMLException):
    """Internal server processing error"""
    pass
```

**Common scenarios:**
- Unexpected runtime errors during prediction
- Resource allocation failures
- Database connection issues
- Model loading failures

### `ServiceUnavailable`
Triggered when incoming requests exceed server capacity.

```python
class ServiceUnavailable(BentoMLException):
    """Service temporarily unavailable"""
    pass
```

**Common scenarios:**
- Request queue is full
- Server overloaded with concurrent requests
- Resource limits exceeded
- Temporary service maintenance

## Resource and Configuration Exceptions

### `NotFound`
Indicates that a specified resource or name cannot be located.

```python
class NotFound(BentoMLException):
    """Resource not found"""
    pass
```

**Common scenarios:**
- Model not found in model store
- Bento package not found
- API endpoint not found
- File or directory not found

**Examples:**
```python
# Model not found
raise NotFound(f"Model 'fraud_detector:v1.0' not found in model store")

# Bento not found
raise NotFound(f"Bento 'my_service:latest' not found")
```

### `BentoMLConfigException`
Raised for misconfiguration or missing required configurations.

```python
class BentoMLConfigException(BentoMLException):
    """Configuration error"""
    pass
```

**Common scenarios:**
- Invalid configuration file format
- Missing required configuration parameters
- Conflicting configuration values
- Invalid environment variable values

**Examples:**
```python
# Invalid config format
raise BentoMLConfigException("Invalid YAML format in bentofile.yaml")

# Missing required config
raise BentoMLConfigException("Missing required configuration: 'service'")
```

## Exception Hierarchy

```
BentoMLException
├── RemoteException
├── InvalidArgument
├── MissingDependencyException
├── InternalServerError
├── NotFound
├── BadInput
├── ServiceUnavailable
├── BentoMLConfigException
└── ImportServiceError
```

## Error Handling Best Practices

### 1. Catch Specific Exceptions
```python
try:
    model = bentoml.load_model("my_model:latest")
except NotFound:
    print("Model not found, using default model")
    model = load_default_model()
except MissingDependencyException as e:
    print(f"Missing dependency: {e}")
    install_dependencies()
```

### 2. Graceful Degradation
```python
try:
    result = expensive_prediction(input_data)
except ServiceUnavailable:
    # Fall back to simpler model
    result = simple_prediction(input_data)
```

### 3. Custom Error Messages
```python
try:
    validate_input(data)
except InvalidArgument as e:
    raise InvalidArgument(f"Validation failed for input data: {e}")
```

### 4. Logging and Monitoring
```python
import logging

try:
    process_request(request)
except BentoMLException as e:
    logging.error(f"BentoML error: {e}", exc_info=True)
    # Send to monitoring system
    monitor.record_error(e)
    raise
```

### 5. API Error Responses
```python
from bentoml import Service

@service.api
def predict(data):
    try:
        return model.predict(data)
    except BadInput as e:
        return {"error": "Invalid input", "details": str(e)}, 400
    except InternalServerError as e:
        return {"error": "Internal error", "details": str(e)}, 500
```

## Common Error Scenarios and Solutions

### Model Loading Issues
```python
try:
    model = bentoml.load_model("my_model:latest")
except NotFound:
    # Check available models
    models = bentoml.list_models()
    print(f"Available models: {models}")
except MissingDependencyException as e:
    print(f"Install missing dependency: {e}")
```

### Service Import Failures
```python
try:
    from my_service import MyService
except ImportServiceError as e:
    print(f"Service import failed: {e}")
    # Check service file syntax and dependencies
```

### Configuration Problems
```python
try:
    config = load_bentoml_config()
except BentoMLConfigException as e:
    print(f"Configuration error: {e}")
    # Provide default configuration or guide user to fix
```