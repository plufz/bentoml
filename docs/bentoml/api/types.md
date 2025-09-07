# BentoML Types

This module contains type definitions and data structures used throughout BentoML for model signatures, configurations, and API specifications.

## ModelSignature

### Class Definition
```python
class ModelSignature(
    batchable: bool = False, 
    batch_dim: Tuple[int, int] = (0, 0), 
    input_spec: Any = None, 
    output_spec: Any = None
)
```

### Description
A model signature represents a method on a model object that can be called. It provides configuration for how model predictions are processed, particularly for batching operations.

### Parameters

#### `batchable`
- **Type**: `bool`
- **Default**: `False`
- **Purpose**: Determines whether multiple API calls to the predict method should be batched by the BentoML runner
- **Usage**: Set to `True` to enable automatic batching for improved throughput

#### `batch_dim`
- **Type**: `Tuple[int, int]`
- **Default**: `(0, 0)`
- **Purpose**: Specifies the dimension(s) for batching input and output data
- **Format**: `(input_batch_dim, output_batch_dim)`
- **Usage**: Configure which dimensions should be used for batching

#### `input_spec` and `output_spec`
- **Type**: `Any`
- **Default**: `None`
- **Status**: Reserved for future use
- **Purpose**: Will define input/output specifications for validation

### Example Usage

#### Basic Batching Configuration
```python
import bentoml

# Save model with batching enabled
bentoml.pytorch.save_model(
    "demo_model", 
    model, 
    signatures={
        "predict": {
            "batchable": True, 
            "batch_dim": 0  # Batch on first dimension
        }
    }
)
```

#### Advanced Batching Configuration
```python
# Different batch dimensions for input and output
signature = ModelSignature(
    batchable=True,
    batch_dim=(0, 0)  # Both input and output batch on dimension 0
)

bentoml.sklearn.save_model(
    "advanced_model",
    model,
    signatures={"predict": signature}
)
```

#### Multiple Methods with Different Signatures
```python
signatures = {
    "predict": ModelSignature(batchable=True, batch_dim=(0, 0)),
    "predict_proba": ModelSignature(batchable=True, batch_dim=(0, 1)),
    "transform": ModelSignature(batchable=False)
}

bentoml.sklearn.save_model("multi_method_model", model, signatures=signatures)
```

### Dictionary Format
ModelSignatures can also be defined using dictionaries with corresponding keys:

```python
signature_dict = {
    "batchable": True,
    "batch_dim": (0, 0),
    "input_spec": None,
    "output_spec": None
}

bentoml.pytorch.save_model("model", model, signatures={"predict": signature_dict})
```

### Batching Behavior

#### When `batchable=True`
- Multiple concurrent requests are automatically batched
- Improves throughput for high-concurrency scenarios
- Reduces per-request overhead
- Batches are processed together on the specified dimension

#### When `batchable=False` (Default)
- Each request is processed individually
- Lower latency for single requests
- Simpler debugging and error handling
- Recommended for low-concurrency scenarios

### Batch Dimension Configuration

The `batch_dim` parameter allows flexible batching configurations:

#### Single Integer (Shorthand)
```python
ModelSignature(batchable=True, batch_dim=0)  # Equivalent to (0, 0)
```

#### Tuple for Different Input/Output Dimensions
```python
# Input batches on dim 0, output on dim 1
ModelSignature(batchable=True, batch_dim=(0, 1))
```

### Use Cases

#### Image Classification
```python
# Images typically batch on the first dimension
ModelSignature(batchable=True, batch_dim=0)
```

#### Time Series Prediction
```python
# Time series might batch on different dimensions
ModelSignature(batchable=True, batch_dim=(0, 0))
```

#### Text Processing
```python
# Text embeddings often batch on the sequence dimension
ModelSignature(batchable=True, batch_dim=(0, 0))
```

## ModelSignatureDict

A type alias for dictionary-based model signature definitions. This provides an alternative to the `ModelSignature` class for cases where dictionary format is preferred.

```python
ModelSignatureDict = Dict[str, Any]

# Example usage
signature: ModelSignatureDict = {
    "batchable": True,
    "batch_dim": (0, 0)
}
```

## Best Practices

### 1. Enable Batching for High-Throughput Services
```python
# For services expecting many concurrent requests
ModelSignature(batchable=True, batch_dim=0)
```

### 2. Disable Batching for Low-Latency Requirements
```python
# For services requiring immediate response
ModelSignature(batchable=False)
```

### 3. Test Batch Dimensions Carefully
```python
# Ensure your model handles the specified batch dimensions correctly
signature = ModelSignature(batchable=True, batch_dim=(0, 0))
# Test with batched inputs to verify correctness
```

### 4. Use Consistent Signatures Across Methods
```python
# Keep similar methods with similar batching behavior
signatures = {
    "predict": ModelSignature(batchable=True, batch_dim=0),
    "predict_proba": ModelSignature(batchable=True, batch_dim=0)
}
```