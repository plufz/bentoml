# BentoML Input/Output Types

Comprehensive guide to handling data types in BentoML services.

## Overview

BentoML provides flexible input/output type handling with automatic serialization, validation, and documentation generation. Types are defined using Python type annotations and Pydantic models.

## Supported Types

### Standard Python Types

```python
@bentoml.api
def basic_types(
    self, 
    text: str,
    number: int, 
    decimal: float,
    flag: bool,
    items: list,
    data: dict
) -> dict:
    return {"processed": True}
```

### Pydantic Models

```python
from pydantic import BaseModel, Field

class InputModel(BaseModel):
    name: str = Field(..., max_length=100)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

@bentoml.api
def structured_input(self, data: InputModel) -> dict:
    return {"name": data.name, "age": data.age}
```

## File Handling

### File Paths

```python
from pathlib import Path
from typing import Annotated
from bentoml._internal.types import ContentType

@bentoml.api
def process_file(
    self, 
    file: Annotated[Path, ContentType('image/jpeg')]
) -> str:
    # File is automatically saved to temporary path
    with open(file, 'rb') as f:
        content = f.read()
    return f"Processed {len(content)} bytes"
```

### Bytes Input

```python
@bentoml.api
def process_bytes(self, data: bytes) -> dict:
    # Raw bytes handling
    return {"size": len(data), "type": "bytes"}
```

### Multiple Content Types

```python
@bentoml.api
def flexible_file(
    self,
    file: Annotated[Path, ContentType(['image/jpeg', 'image/png', 'audio/wav'])]
) -> dict:
    # Accepts multiple file types
    return {"file_path": str(file)}
```

## Machine Learning Types

### NumPy Arrays

```python
import numpy as np
from typing import Annotated
from bentoml.validators import Shape, DType

@bentoml.api
def numpy_input(
    self,
    array: Annotated[np.ndarray, Shape((28, 28)), DType("float32")]
) -> np.ndarray:
    # Validates shape and dtype
    return array * 2
```

### Pandas DataFrames

```python
import pandas as pd
from bentoml.validators import Columns

@bentoml.api
def dataframe_input(
    self,
    df: Annotated[pd.DataFrame, Columns(['feature1', 'feature2', 'target'])]
) -> pd.DataFrame:
    # Validates required columns
    return df.describe()
```

### PyTorch Tensors

```python
import torch
from typing import Annotated

@bentoml.api
def tensor_input(
    self,
    tensor: Annotated[torch.Tensor, Shape((None, 3, 224, 224))]
) -> torch.Tensor:
    # Process tensor
    return tensor.mean(dim=1)
```

## Advanced Input/Output Patterns

### Compound Inputs

```python
class CompoundInput(BaseModel):
    text: str
    image: bytes
    metadata: dict

@bentoml.api
def compound_processing(
    self, 
    request: CompoundInput
) -> dict:
    return {
        "text_length": len(request.text),
        "image_size": len(request.image),
        "metadata_keys": list(request.metadata.keys())
    }
```

### List Inputs/Outputs

```python
@bentoml.api
def batch_processing(
    self, 
    texts: list[str]
) -> list[dict]:
    return [{"text": t, "length": len(t)} for t in texts]
```

### Optional Parameters

```python
from typing import Optional

@bentoml.api
def optional_params(
    self,
    required: str,
    optional: Optional[str] = None,
    default_value: int = 42
) -> dict:
    return {
        "required": required,
        "optional": optional,
        "default": default_value
    }
```

### Root Input (No JSON Wrapper)

```python
@bentoml.api
def root_input(self, data: str) -> str:
    # Accepts raw string without JSON wrapper
    return data.upper()
```

## Validation and Constraints

### String Validation

```python
class TextInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(..., regex=r'^(news|blog|article)$')

@bentoml.api
def validate_text(self, input: TextInput) -> dict:
    return {"valid": True, "category": input.category}
```

### Numeric Validation

```python
class NumericInput(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)  # Between 0 and 1
    count: int = Field(..., gt=0)              # Greater than 0

@bentoml.api
def validate_numbers(self, input: NumericInput) -> dict:
    return {"normalized": input.value, "count": input.count}
```

### Array Validation

```python
from bentoml.validators import Shape, DType

@bentoml.api
def validate_array(
    self,
    data: Annotated[np.ndarray, Shape((None, 784)), DType("float32")]
) -> dict:
    # Validates shape (N, 784) and float32 dtype
    return {"shape": data.shape, "dtype": str(data.dtype)}
```

## Content Type Validation

### Image Files

```python
@bentoml.api
def process_image(
    self,
    image: Annotated[Path, ContentType('image/*')]
) -> dict:
    # Accepts any image format
    return {"file_size": image.stat().st_size}
```

### Specific Formats

```python
@bentoml.api
def audio_processing(
    self,
    audio: Annotated[Path, ContentType(['audio/wav', 'audio/mp3'])]
) -> dict:
    return {"format": "supported"}
```

## Response Types

### Structured Responses

```python
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

@bentoml.api
def structured_response(self, input: str) -> PredictionResponse:
    return PredictionResponse(
        prediction=0.95,
        confidence=0.87,
        model_version="v1.0"
    )
```

### File Responses

```python
@bentoml.api
def generate_file(self, prompt: str) -> Path:
    # Generate and return file
    output_path = Path("/tmp/generated.txt")
    with open(output_path, 'w') as f:
        f.write(f"Generated from: {prompt}")
    return output_path
```

## Error Handling

### Custom Validation

```python
from pydantic import validator

class ValidatedInput(BaseModel):
    email: str
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

@bentoml.api
def validated_endpoint(self, input: ValidatedInput) -> dict:
    return {"valid_email": input.email}
```

### Error Responses

```python
@bentoml.api
def error_handling(self, data: str) -> dict:
    try:
        # Process data
        result = self.process(data)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": "Internal server error"}
```

## Best Practices

1. **Use Type Annotations**: Always specify input/output types
2. **Validate Early**: Use Pydantic models for complex validation
3. **Handle Errors Gracefully**: Return structured error responses
4. **Document Constraints**: Use Field descriptions and examples
5. **Test Edge Cases**: Validate with various input formats
6. **Consider Performance**: Use appropriate data types for your use case

## Example: Complete File Upload Handler

```python
from pathlib import Path
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from bentoml._internal.types import ContentType

class ProcessingOptions(BaseModel):
    quality: Optional[str] = Field("high", regex=r'^(low|medium|high)$')
    format: Optional[str] = Field("png", regex=r'^(png|jpg|webp)$')

class ProcessingResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: dict

@bentoml.api
def process_image_upload(
    self,
    image: Annotated[Path, ContentType(['image/jpeg', 'image/png'])],
    options: ProcessingOptions
) -> ProcessingResponse:
    try:
        # Process the uploaded image
        processed_path = self.image_processor.process(
            image, 
            quality=options.quality,
            output_format=options.format
        )
        
        return ProcessingResponse(
            success=True,
            output_path=str(processed_path),
            metadata={
                "input_size": image.stat().st_size,
                "format": options.format,
                "quality": options.quality
            }
        )
        
    except Exception as e:
        return ProcessingResponse(
            success=False,
            error=str(e),
            metadata={}
        )
```

This example demonstrates file upload handling, validation, structured responses, and error handling in a production-ready pattern.