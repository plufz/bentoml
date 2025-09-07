# BentoML PyTorch Integration

Comprehensive API reference for integrating PyTorch models with BentoML.

## Overview

BentoML provides seamless integration with PyTorch, supporting model saving, loading, and deployment with optimized performance and device management.

## Core Functions

### save_model()

Save a PyTorch model to BentoML's model store.

```python
bentoml.pytorch.save_model(
    name: str,
    model: torch.nn.Module,
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification
- `model` (torch.nn.Module): PyTorch model to save
- `signatures` (dict, optional): Model signature configuration for batching
- `labels` (dict, optional): Labels for model organization
- `custom_objects` (dict, optional): Additional objects to save with model
- `metadata` (dict, optional): Custom metadata

**Returns:** `bentoml.Tag` with model name and version

#### Basic Usage

```python
import torch
import bentoml

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train your model
model = SimpleModel()
# ... training code ...

# Save model
tag = bentoml.pytorch.save_model("simple_model", model)
print(f"Model saved: {tag}")
```

#### Advanced Configuration

```python
# Save with comprehensive configuration
tag = bentoml.pytorch.save_model(
    name="advanced_pytorch_model",
    model=trained_model,
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "stage": "production",
        "framework": "pytorch",
        "task": "classification"
    },
    metadata={
        "accuracy": 0.95,
        "f1_score": 0.92,
        "training_dataset": "imagenet",
        "epochs": 100
    },
    custom_objects={
        "tokenizer": tokenizer,
        "preprocessor": preprocessor
    }
)
```

#### Model with Custom Objects

```python
import torchvision.transforms as transforms

# Save model with preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

tag = bentoml.pytorch.save_model(
    name="resnet_classifier",
    model=resnet_model,
    custom_objects={
        "transform": transform,
        "class_names": ["cat", "dog", "bird"]
    }
)
```

### load_model()

Load a saved PyTorch model from BentoML's model store.

```python
bentoml.pytorch.load_model(
    tag: str,
    device_id: str = None
) -> torch.nn.Module
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)
- `device_id` (str, optional): Target device ("cpu", "cuda:0", etc.)

**Returns:** `torch.nn.Module` instance

#### Basic Loading

```python
# Load latest version
model = bentoml.pytorch.load_model("simple_model:latest")

# Load specific version
model = bentoml.pytorch.load_model("simple_model:v1.2.0")

# Load to specific device
model = bentoml.pytorch.load_model("simple_model:latest", device_id="cuda:0")
```

#### Device Management

```python
import torch

# Load to CPU
cpu_model = bentoml.pytorch.load_model("model_name", device_id="cpu")

# Load to GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = bentoml.pytorch.load_model("model_name", device_id=device)

# Load to specific GPU
multi_gpu_model = bentoml.pytorch.load_model("model_name", device_id="cuda:1")
```

### get()

Get a BentoML model reference without loading into memory.

```python
model_ref = bentoml.pytorch.get("model_name:latest")
# Use model_ref.path for model files
# Use model_ref.info for metadata
```

## Service Integration

### Basic Service

```python
import bentoml
import torch

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
class PyTorchClassifier:
    def __init__(self):
        # Load model during service initialization
        self.model = bentoml.pytorch.load_model("classifier:latest")
        self.model.eval()  # Set to evaluation mode
        
        # Load custom objects if saved
        model_ref = bentoml.pytorch.get("classifier:latest")
        self.transform = model_ref.custom_objects.get("transform")
        self.class_names = model_ref.custom_objects.get("class_names", [])
    
    @bentoml.api
    def predict(self, input_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            # Apply preprocessing if available
            if self.transform:
                input_tensor = self.transform(input_tensor)
            
            # Ensure correct device
            if next(self.model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            
            # Run inference
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return {
                "class_id": predicted_class.item(),
                "class_name": self.class_names[predicted_class.item()] if self.class_names else None,
                "confidence": probabilities.max().item(),
                "probabilities": probabilities.tolist()
            }
```

### Batch Processing Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class BatchPyTorchService:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("batch_model:latest")
        self.model.eval()
    
    @bentoml.api(
        batch=True,
        max_batch_size=32,
        max_latency_ms=100
    )
    def batch_predict(self, input_tensors: list[torch.Tensor]) -> list[dict]:
        # Stack tensors for batch processing
        batch_tensor = torch.stack(input_tensors)
        
        with torch.no_grad():
            batch_output = self.model(batch_tensor)
            batch_probabilities = torch.softmax(batch_output, dim=1)
        
        # Convert batch results to individual predictions
        results = []
        for i in range(len(input_tensors)):
            results.append({
                "prediction": batch_output[i].tolist(),
                "confidence": batch_probabilities[i].max().item()
            })
        
        return results
```

### Multi-Model Service

```python
@bentoml.service(
    resources={"gpu": "2", "memory": "12Gi"}
)
class MultiPyTorchService:
    def __init__(self):
        # Load multiple PyTorch models
        self.classifier = bentoml.pytorch.load_model("classifier:latest")
        self.detector = bentoml.pytorch.load_model("detector:latest")
        self.generator = bentoml.pytorch.load_model("generator:latest")
        
        # Set all models to evaluation mode
        self.classifier.eval()
        self.detector.eval()  
        self.generator.eval()
    
    @bentoml.api
    def classify(self, image: torch.Tensor) -> dict:
        with torch.no_grad():
            output = self.classifier(image.unsqueeze(0))
            return {"class": output.argmax().item()}
    
    @bentoml.api
    def detect_objects(self, image: torch.Tensor) -> dict:
        with torch.no_grad():
            detections = self.detector(image.unsqueeze(0))
            return {"detections": detections.tolist()}
    
    @bentoml.api
    def generate_image(self, noise: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            generated = self.generator(noise)
            return generated.squeeze(0)
```

## Advanced Features

### TorchScript Integration

```python
import torch.jit

# Save TorchScript model
scripted_model = torch.jit.script(model)
tag = bentoml.pytorch.save_model("scripted_model", scripted_model)

# Load and use TorchScript model
@bentoml.service
class TorchScriptService:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("scripted_model:latest")
    
    @bentoml.api
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.model(input_data)
```

### State Dict Handling

```python
# Save only state dict with model architecture
def save_with_state_dict():
    model_state = {
        'state_dict': model.state_dict(),
        'architecture': 'ResNet50',
        'num_classes': 1000
    }
    
    tag = bentoml.pytorch.save_model(
        "model_with_state",
        model,
        custom_objects={'model_config': model_state}
    )

# Load and reconstruct model
@bentoml.service
class StateDictService:
    def __init__(self):
        model_ref = bentoml.pytorch.get("model_with_state:latest")
        config = model_ref.custom_objects['model_config']
        
        # Reconstruct model architecture
        self.model = create_model_from_config(config)
        self.model.load_state_dict(config['state_dict'])
        self.model.eval()
```

### Dynamic Model Loading

```python
@bentoml.service
class DynamicPyTorchService:
    def __init__(self):
        self.models = {}
    
    def load_model_on_demand(self, model_name: str):
        if model_name not in self.models:
            self.models[model_name] = bentoml.pytorch.load_model(f"{model_name}:latest")
            self.models[model_name].eval()
        return self.models[model_name]
    
    @bentoml.api
    def predict_with_model(self, model_name: str, input_data: torch.Tensor) -> dict:
        model = self.load_model_on_demand(model_name)
        
        with torch.no_grad():
            output = model(input_data)
            return {
                "model_used": model_name,
                "prediction": output.tolist()
            }
```

## Best Practices

### 1. Device Management

```python
@bentoml.service(resources={"gpu": "1"})
class DeviceAwareService:
    def __init__(self):
        # Determine best available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model to specific device
        self.model = bentoml.pytorch.load_model("model:latest", device_id=str(self.device))
        self.model.eval()
    
    @bentoml.api
    def predict(self, input_data: torch.Tensor) -> dict:
        # Ensure input is on correct device
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_data)
            return {"prediction": output.cpu().tolist()}  # Move back to CPU for JSON
```

### 2. Memory Management

```python
@bentoml.service
class MemoryEfficientService:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("large_model:latest")
        self.model.eval()
    
    @bentoml.api
    def predict(self, input_data: torch.Tensor) -> dict:
        with torch.no_grad():
            # Use torch.cuda.empty_cache() for GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            output = self.model(input_data)
            
            # Clean up intermediate tensors
            del input_data
            
            result = {"prediction": output.tolist()}
            del output
            
            return result
```

### 3. Error Handling

```python
@bentoml.service
class RobustPyTorchService:
    def __init__(self):
        try:
            self.model = bentoml.pytorch.load_model("model:latest")
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    @bentoml.api
    def predict(self, input_data: torch.Tensor) -> dict:
        if not self.model_loaded:
            return {"error": "Model not available"}
        
        try:
            # Validate input shape
            if input_data.dim() != 4:  # Expecting batch of images
                return {"error": f"Expected 4D tensor, got {input_data.dim()}D"}
            
            with torch.no_grad():
                output = self.model(input_data)
                return {
                    "success": True,
                    "prediction": output.tolist()
                }
                
        except RuntimeError as e:
            logger.error(f"PyTorch runtime error: {e}")
            return {"error": "Inference failed", "details": str(e)}
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": "Unexpected error occurred"}
```

### 4. Model Versioning

```python
# Save with semantic versioning
def save_versioned_model(model, version="1.0.0"):
    tag = bentoml.pytorch.save_model(
        name="production_model",
        model=model,
        labels={
            "version": version,
            "stage": "production"
        },
        metadata={
            "created_at": datetime.now().isoformat(),
            "pytorch_version": torch.__version__
        }
    )
    return tag

# Load specific version
@bentoml.service
class VersionedService:
    def __init__(self):
        # Load latest production model
        models = bentoml.models.list()
        production_models = [m for m in models if m.labels.get("stage") == "production"]
        latest_prod = max(production_models, key=lambda x: x.creation_time)
        
        self.model = bentoml.pytorch.load_model(latest_prod.tag)
        self.model.eval()
```

## Examples

### Image Classification

```python
import torchvision.transforms as transforms
from PIL import Image

@bentoml.service
class ImageClassifier:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("resnet_classifier:latest")
        self.model.eval()
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @bentoml.api
    def classify_image(self, image: Image.Image) -> dict:
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        return {
            "top_predictions": [
                {
                    "class_id": idx.item(),
                    "confidence": prob.item()
                }
                for idx, prob in zip(top5_idx[0], top5_prob[0])
            ]
        }
```

### Text Generation

```python
@bentoml.service
class TextGenerator:
    def __init__(self):
        self.model = bentoml.pytorch.load_model("gpt_model:latest")
        self.model.eval()
        
        model_ref = bentoml.pytorch.get("gpt_model:latest")
        self.tokenizer = model_ref.custom_objects["tokenizer"]
    
    @bentoml.api
    def generate_text(self, prompt: str, max_length: int = 100) -> dict:
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # Generate text
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "length": len(generated_text)
        }
```

For more PyTorch examples, visit the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/).