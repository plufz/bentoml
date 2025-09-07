# BentoML ONNX Integration

Comprehensive API reference for integrating ONNX models with BentoML.

## Overview

BentoML provides robust integration with ONNX (Open Neural Network Exchange), supporting ONNX Runtime for high-performance inference across different hardware platforms and optimization features.

## Core Functions

### save_model()

Save an ONNX model to BentoML's model store.

```python
bentoml.onnx.save_model(
    name: str,
    model: Union[str, Path, onnx.ModelProto],
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification
- `model` (Union[str, Path, onnx.ModelProto]): ONNX model file path or ModelProto
- `signatures` (dict, optional): Model signature configuration (default: `{"run": {"batchable": False}}`)
- `labels` (dict, optional): Labels for model organization
- `custom_objects` (dict, optional): Additional objects to save
- `metadata` (dict, optional): Custom metadata

**Returns:** `bentoml.Tag` with model name and version

#### Basic Usage

```python
import bentoml
import onnx

# Save from file path
tag = bentoml.onnx.save_model("onnx_classifier", "./model.onnx")
print(f"Model saved: {tag}")

# Save from ONNX ModelProto
model_proto = onnx.load("./model.onnx")
tag = bentoml.onnx.save_model("onnx_classifier_proto", model_proto)
```

#### Advanced Configuration

```python
# Save with comprehensive configuration
tag = bentoml.onnx.save_model(
    name="production_onnx_model",
    model="./optimized_model.onnx",
    signatures={
        "run": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "stage": "production",
        "optimization": "tensorrt",
        "precision": "fp16"
    },
    metadata={
        "accuracy": 0.95,
        "model_size_mb": 45.2,
        "input_shape": [1, 3, 224, 224],
        "output_classes": 1000,
        "opset_version": 11,
        "converted_from": "pytorch"
    },
    custom_objects={
        "class_names": class_names_list,
        "preprocessing_params": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
)
```

### load_model()

Load a saved ONNX model from BentoML's model store.

```python
bentoml.onnx.load_model(
    tag: str,
    providers: List[str] = None
) -> onnxruntime.InferenceSession
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)
- `providers` (List[str], optional): ONNX Runtime execution providers

**Returns:** `onnxruntime.InferenceSession` for inference

#### Basic Loading

```python
import onnxruntime

# Load with default CPU provider
session = bentoml.onnx.load_model("onnx_classifier:latest")

# Load with specific providers
session = bentoml.onnx.load_model(
    "onnx_classifier:latest",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

#### Provider Configuration

```python
# Check available providers
print("Available providers:", onnxruntime.get_available_providers())

# Load with optimized providers
gpu_providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
        "cudnn_conv_algo_search": "EXHAUSTIVE"
    }),
    "CPUExecutionProvider"
]

session = bentoml.onnx.load_model("onnx_model:latest", providers=gpu_providers)
```

### get()

Get a BentoML model reference without loading into memory.

```python
model_ref = bentoml.onnx.get("model_name:latest")
# Access model path, metadata, custom objects
```

## Service Integration

### Image Classification Service

```python
import bentoml
import numpy as np
from PIL import Image
from typing import List
import onnxruntime

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
class ONNXImageClassifier:
    def __init__(self):
        # Load ONNX model
        self.session = bentoml.onnx.load_model(
            "image_classifier:latest",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Load custom objects
        model_ref = bentoml.onnx.get("image_classifier:latest")
        self.class_names = model_ref.custom_objects.get("class_names", [])
        self.preprocessing_params = model_ref.custom_objects.get("preprocessing_params", {})
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        # Resize image
        target_size = (self.input_shape[2], self.input_shape[3])  # H, W
        image = image.convert('RGB').resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Apply normalization if available
        if self.preprocessing_params:
            mean = np.array(self.preprocessing_params.get("mean", [0, 0, 0]))
            std = np.array(self.preprocessing_params.get("std", [1, 1, 1]))
            image_array = (image_array - mean) / std
        
        # Add batch dimension and transpose to NCHW
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    @bentoml.api
    def classify_image(self, image: Image.Image) -> dict:
        # Preprocess image
        input_data = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Get top prediction
        predicted_class = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions)[::-1][:5]
        top5_predictions = [
            {
                "class_id": int(idx),
                "class_name": self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                "confidence": float(predictions[idx])
            }
            for idx in top5_indices
        ]
        
        return {
            "predicted_class": int(predicted_class),
            "predicted_class_name": self.class_names[predicted_class] if predicted_class < len(self.class_names) else None,
            "confidence": confidence,
            "top5_predictions": top5_predictions
        }
    
    @bentoml.api
    def classify_batch(self, images: List[Image.Image]) -> List[dict]:
        # Batch processing
        results = []
        
        for image in images:
            try:
                result = self.classify_image(image)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
```

### Text Processing Service

```python
@bentoml.service
class ONNXTextProcessor:
    def __init__(self):
        self.session = bentoml.onnx.load_model("text_classifier:latest")
        
        # Get input/output specifications
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load tokenizer and vocabulary
        model_ref = bentoml.onnx.get("text_classifier:latest")
        self.vocab = model_ref.custom_objects.get("vocab", {})
        self.max_length = model_ref.metadata.get("max_length", 128)
        self.class_names = model_ref.custom_objects.get("class_names", [])
    
    def tokenize_text(self, text: str) -> np.ndarray:
        # Simple tokenization (in practice, use proper tokenizer)
        tokens = text.lower().split()
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, 0) for token in tokens]  # 0 for unknown
        
        # Pad or truncate to max_length
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        # Convert to numpy array with batch dimension
        return np.array([token_ids], dtype=np.int64)
    
    @bentoml.api
    def classify_text(self, text: str) -> dict:
        # Tokenize input
        input_ids = self.tokenize_text(text)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_ids})
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        return {
            "text": text,
            "predicted_class": int(predicted_class),
            "predicted_class_name": self.class_names[predicted_class] if predicted_class < len(self.class_names) else None,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }
```

### Object Detection Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "6Gi"}
)
class ONNXObjectDetector:
    def __init__(self):
        self.session = bentoml.onnx.load_model(
            "object_detector:latest",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Get model I/O info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Load detection parameters
        model_ref = bentoml.onnx.get("object_detector:latest")
        self.class_names = model_ref.custom_objects.get("class_names", [])
        self.confidence_threshold = model_ref.metadata.get("confidence_threshold", 0.5)
        self.nms_threshold = model_ref.metadata.get("nms_threshold", 0.4)
    
    def preprocess_image(self, image: Image.Image) -> tuple:
        # Store original dimensions
        original_width, original_height = image.size
        
        # Resize to model input size
        target_height, target_width = self.input_shape[2], self.input_shape[3]
        resized_image = image.resize((target_width, target_height))
        
        # Convert to numpy array
        image_array = np.array(resized_image, dtype=np.float32) / 255.0
        
        # Transpose and add batch dimension
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array, (original_width, original_height)
    
    def postprocess_detections(self, outputs, original_size):
        # Process ONNX model outputs (this depends on your specific model)
        # Example for YOLO-style outputs
        boxes = outputs[0]  # [batch, num_boxes, 4]
        scores = outputs[1]  # [batch, num_boxes, num_classes]
        
        # Apply confidence threshold and NMS
        detections = []
        
        for i in range(boxes.shape[1]):
            box = boxes[0, i]
            class_scores = scores[0, i]
            
            max_score = np.max(class_scores)
            if max_score > self.confidence_threshold:
                class_id = np.argmax(class_scores)
                
                # Scale box coordinates to original image size
                x1, y1, x2, y2 = box
                x1 = int(x1 * original_size[0])
                y1 = int(y1 * original_size[1])
                x2 = int(x2 * original_size[0])
                y2 = int(y2 * original_size[1])
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                    "confidence": float(max_score)
                })
        
        return detections
    
    @bentoml.api
    def detect_objects(self, image: Image.Image) -> dict:
        # Preprocess image
        input_data, original_size = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocess detections
        detections = self.postprocess_detections(outputs, original_size)
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_size": original_size,
            "confidence_threshold": self.confidence_threshold
        }
```

## Advanced Features

### Model Optimization

```python
# Save optimized ONNX model
import onnxruntime as ort

# Create optimization configuration
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "./optimized_model.onnx"

# Save with optimization metadata
tag = bentoml.onnx.save_model(
    "optimized_model",
    "./model.onnx",
    metadata={
        "optimization_level": "ORT_ENABLE_ALL",
        "optimized": True,
        "original_size_mb": 120.5,
        "optimized_size_mb": 98.2
    }
)
```

### Dynamic Input Shapes

```python
@bentoml.service
class DynamicONNXService:
    def __init__(self):
        self.session = bentoml.onnx.load_model("dynamic_model:latest")
        
        # Check if model supports dynamic shapes
        input_shape = self.session.get_inputs()[0].shape
        self.dynamic_dims = [i for i, dim in enumerate(input_shape) if isinstance(dim, str)]
    
    @bentoml.api
    def process_variable_input(self, data: List[List[float]]) -> dict:
        # Handle variable batch size
        input_array = np.array(data, dtype=np.float32)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        outputs = self.session.run([output_name], {input_name: input_array})
        
        return {
            "input_shape": input_array.shape,
            "output_shape": outputs[0].shape,
            "results": outputs[0].tolist()
        }
```

### Multi-Model Inference

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class MultiONNXService:
    def __init__(self):
        # Load multiple ONNX models
        self.classifier = bentoml.onnx.load_model("classifier:latest")
        self.detector = bentoml.onnx.load_model("detector:latest")
        self.segmentor = bentoml.onnx.load_model("segmentor:latest")
        
        # Store input/output names for each model
        self.model_specs = {
            "classifier": {
                "input": self.classifier.get_inputs()[0].name,
                "output": self.classifier.get_outputs()[0].name
            },
            "detector": {
                "input": self.detector.get_inputs()[0].name,
                "outputs": [out.name for out in self.detector.get_outputs()]
            },
            "segmentor": {
                "input": self.segmentor.get_inputs()[0].name,
                "output": self.segmentor.get_outputs()[0].name
            }
        }
    
    @bentoml.api
    def comprehensive_analysis(self, image: Image.Image) -> dict:
        # Preprocess image for all models
        image_array = self.preprocess_for_classification(image)
        
        results = {}
        
        # Classification
        try:
            cls_output = self.classifier.run(
                [self.model_specs["classifier"]["output"]], 
                {self.model_specs["classifier"]["input"]: image_array}
            )
            results["classification"] = {
                "predicted_class": int(np.argmax(cls_output[0])),
                "confidence": float(np.max(cls_output[0]))
            }
        except Exception as e:
            results["classification"] = {"error": str(e)}
        
        # Object Detection
        try:
            det_image_array = self.preprocess_for_detection(image)
            det_outputs = self.detector.run(
                self.model_specs["detector"]["outputs"],
                {self.model_specs["detector"]["input"]: det_image_array}
            )
            results["detection"] = self.postprocess_detection(det_outputs)
        except Exception as e:
            results["detection"] = {"error": str(e)}
        
        # Segmentation
        try:
            seg_image_array = self.preprocess_for_segmentation(image)
            seg_output = self.segmentor.run(
                [self.model_specs["segmentor"]["output"]],
                {self.model_specs["segmentor"]["input"]: seg_image_array}
            )
            results["segmentation"] = self.postprocess_segmentation(seg_output[0])
        except Exception as e:
            results["segmentation"] = {"error": str(e)}
        
        return results
```

## Performance Optimization

### Provider Optimization

```python
@bentoml.service(resources={"gpu": "1"})
class OptimizedONNXService:
    def __init__(self):
        # Configure optimized providers
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB
                "cudnn_conv_algo_search": "EXHAUSTIVE"
            }),
            ("CPUExecutionProvider", {
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 4
            })
        ]
        
        # Create session with optimizations
        self.session = bentoml.onnx.load_model("optimized_model:latest", providers=providers)
    
    @bentoml.api
    def optimized_inference(self, input_data: List[List[float]]) -> dict:
        input_array = np.array(input_data, dtype=np.float32)
        
        # Run with optimized session
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        outputs = self.session.run([output_name], {input_name: input_array})
        
        return {
            "results": outputs[0].tolist(),
            "provider": self.session.get_providers()[0]
        }
```

### Batch Processing

```python
@bentoml.service
class BatchONNXService:
    def __init__(self):
        self.session = bentoml.onnx.load_model("batch_model:latest")
        self.max_batch_size = 32
        
        # Get dynamic batch info
        input_shape = self.session.get_inputs()[0].shape
        self.batch_dim = 0 if isinstance(input_shape[0], str) else None
    
    @bentoml.api
    def batch_inference(self, inputs: List[List[float]]) -> List[dict]:
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), self.max_batch_size):
            batch = inputs[i:i + self.max_batch_size]
            batch_array = np.array(batch, dtype=np.float32)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            batch_outputs = self.session.run([output_name], {input_name: batch_array})
            
            # Process batch results
            for j, output in enumerate(batch_outputs[0]):
                results.append({
                    "input_index": i + j,
                    "prediction": output.tolist()
                })
        
        return results
```

## Best Practices

### 1. Model Validation

```python
@bentoml.service
class ValidatedONNXService:
    def __init__(self):
        self.session = bentoml.onnx.load_model("validated_model:latest")
        
        # Validate model inputs/outputs
        self.input_spec = {
            "name": self.session.get_inputs()[0].name,
            "shape": self.session.get_inputs()[0].shape,
            "type": self.session.get_inputs()[0].type
        }
        
        self.output_spec = {
            "name": self.session.get_outputs()[0].name,
            "shape": self.session.get_outputs()[0].shape,
            "type": self.session.get_outputs()[0].type
        }
    
    def validate_input(self, input_array: np.ndarray) -> bool:
        # Check shape compatibility (ignoring dynamic dimensions)
        expected_shape = self.input_spec["shape"]
        actual_shape = input_array.shape
        
        if len(expected_shape) != len(actual_shape):
            return False
        
        for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if isinstance(expected, int) and expected != actual:
                return False
        
        return True
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        input_array = np.array(input_data, dtype=np.float32)
        
        # Validate input
        if not self.validate_input(input_array):
            return {
                "error": f"Invalid input shape: {input_array.shape}, expected: {self.input_spec['shape']}"
            }
        
        try:
            outputs = self.session.run(
                [self.output_spec["name"]], 
                {self.input_spec["name"]: input_array}
            )
            
            return {
                "success": True,
                "predictions": outputs[0].tolist(),
                "output_shape": outputs[0].shape
            }
            
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}
```

### 2. Error Handling

```python
@bentoml.service
class RobustONNXService:
    def __init__(self):
        try:
            self.session = bentoml.onnx.load_model("robust_model:latest")
            self.model_loaded = True
            
            # Test model with dummy input
            dummy_input = np.random.randn(1, *self.session.get_inputs()[0].shape[1:]).astype(np.float32)
            input_name = self.session.get_inputs()[0].name
            self.session.run(None, {input_name: dummy_input})
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model_loaded = False
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        if not self.model_loaded:
            return {"error": "Model not available"}
        
        try:
            input_array = np.array(input_data, dtype=np.float32)
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: input_array})
            
            return {
                "success": True,
                "predictions": outputs[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            return {"error": "Prediction failed", "details": str(e)}
```

### 3. Performance Monitoring

```python
import time

@bentoml.service
class MonitoredONNXService:
    def __init__(self):
        self.session = bentoml.onnx.load_model("monitored_model:latest")
        self.inference_times = []
        self.request_count = 0
    
    @bentoml.api
    def predict_with_timing(self, input_data: List[List[float]]) -> dict:
        start_time = time.time()
        
        try:
            input_array = np.array(input_data, dtype=np.float32)
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: input_array})
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.request_count += 1
            
            return {
                "predictions": outputs[0].tolist(),
                "inference_time": inference_time,
                "total_requests": self.request_count,
                "average_time": sum(self.inference_times) / len(self.inference_times),
                "providers": self.session.get_providers()
            }
            
        except Exception as e:
            return {"error": str(e)}
```

For more ONNX examples and optimization techniques, visit the [ONNX Runtime documentation](https://onnxruntime.ai/docs/) and the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/).