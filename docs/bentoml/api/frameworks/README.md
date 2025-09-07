# BentoML Framework APIs

BentoML provides comprehensive integration with multiple machine learning and AI frameworks, enabling seamless model deployment regardless of the underlying framework used for training.

## Supported Frameworks

### Deep Learning Frameworks
- **[PyTorch](pytorch.md)** - Popular deep learning framework with dynamic computation graphs
- **[TensorFlow](tensorflow.md)** - Google's machine learning platform with extensive ecosystem
- **[Flax](flax.md)** - High-performance neural network library for JAX
- **[Keras](keras.md)** - High-level neural networks API

### Traditional ML Frameworks  
- **[Scikit-Learn](scikit-learn.md)** - Simple and efficient tools for data mining and analysis
- **[XGBoost](xgboost.md)** - Optimized distributed gradient boosting library
- **[LightGBM](lightgbm.md)** - Gradient boosting framework with high efficiency
- **[CatBoost](catboost.md)** - Fast and accurate gradient boosting library

### Specialized AI Frameworks
- **[Transformers](transformers.md)** - State-of-the-art Natural Language Processing models
- **[Diffusers](diffusers.md)** - Diffusion models for image and audio generation
- **[EasyOCR](easyocr.md)** - Ready-to-use OCR with broad language support
- **[Detectron](detectron.md)** - Facebook AI Research's object detection platform

### Model Optimization & Serving
- **[ONNX](onnx.md)** - Open standard for machine learning interoperability  
- **[MLflow](mlflow.md)** - Open source platform for ML lifecycle management

### Research & Experimental
- **[Ray](ray.md)** - Distributed computing framework for ML workloads
- **[fast.ai](fastai.md)** - Deep learning for coders library

## Common Integration Patterns

### Basic Model Saving and Loading

```python
import bentoml

# Save model (framework-specific)
bentoml.pytorch.save_model("my_model", trained_model)
bentoml.sklearn.save_model("my_classifier", sklearn_model)
bentoml.transformers.save_model("my_nlp_model", transformer_model)

# Load model in service
@bentoml.service
class MyService:
    def __init__(self):
        self.model = bentoml.pytorch.get("my_model:latest")
    
    @bentoml.api
    def predict(self, input_data):
        return self.model.predict(input_data)
```

### Framework-Specific Configuration

```python
# PyTorch with custom configuration
bentoml.pytorch.save_model(
    "pytorch_model",
    model,
    signatures={"predict": {"batchable": True}},
    metadata={"accuracy": 0.95}
)

# Transformers with tokenizer
bentoml.transformers.save_model(
    "bert_model", 
    model,
    custom_objects={"tokenizer": tokenizer}
)

# Scikit-learn with preprocessing pipeline
bentoml.sklearn.save_model(
    "ml_pipeline",
    pipeline,
    signatures={"predict": {"batchable": True, "batch_dim": 0}}
)
```

### Multi-Framework Services

```python
@bentoml.service
class MultiFrameworkService:
    def __init__(self):
        self.pytorch_model = bentoml.pytorch.get("vision_model:latest")
        self.sklearn_model = bentoml.sklearn.get("classifier:latest") 
        self.transformers_model = bentoml.transformers.get("nlp_model:latest")
    
    @bentoml.api
    def vision_classify(self, image: Image) -> dict:
        features = self.pytorch_model.extract_features(image)
        prediction = self.sklearn_model.predict([features])
        return {"class": prediction[0]}
    
    @bentoml.api  
    def text_analyze(self, text: str) -> dict:
        result = self.transformers_model.predict(text)
        return {"sentiment": result}
```

## Integration Features

### Universal Capabilities
- **Model Versioning** - Automatic versioning for all frameworks
- **Metadata Storage** - Custom metadata and tags
- **Signature Definition** - Batch processing configuration
- **Custom Objects** - Framework-specific extensions
- **Automatic Serialization** - Optimized model storage

### Framework-Specific Features
- **PyTorch**: TorchScript compilation, state dict handling
- **TensorFlow**: SavedModel format, TensorFlow Lite support  
- **Transformers**: Tokenizer integration, pipeline support
- **Scikit-learn**: Pipeline preservation, custom estimators
- **ONNX**: Cross-platform model optimization
- **Diffusers**: Pipeline configuration, scheduler settings

## Model Management

### Saving Models

```python
# Basic saving
bentoml.pytorch.save_model("model_name", model)

# With metadata and configuration
bentoml.pytorch.save_model(
    "advanced_model",
    model,
    labels={"stage": "production", "version": "v1.2"},
    metadata={"accuracy": 0.95, "f1_score": 0.92},
    signatures={"predict": {"batchable": True, "batch_dim": 0}}
)
```

### Loading Models

```python
# Latest version
model = bentoml.pytorch.get("model_name:latest")

# Specific version
model = bentoml.pytorch.get("model_name:v1.2.0")

# With tag
model = bentoml.pytorch.get("model_name", labels={"stage": "production"})
```

### Model Registry Operations

```python
# List models
models = bentoml.models.list()

# Delete model
bentoml.models.delete("model_name:version")

# Export model
bentoml.models.export("model_name:latest", "/path/to/export/")

# Import model  
bentoml.models.import_model("/path/to/model/")
```

## Best Practices

1. **Choose the Right Framework API** - Use the specific framework API for optimal integration
2. **Version Your Models** - Use semantic versioning for model releases
3. **Add Meaningful Metadata** - Include performance metrics and model information
4. **Configure Batch Processing** - Set appropriate batch dimensions for performance
5. **Test Cross-Framework Compatibility** - Verify model behavior across different environments
6. **Use Custom Objects Wisely** - Include necessary preprocessing or postprocessing components
7. **Monitor Model Performance** - Track inference metrics across frameworks

## Framework-Specific Guides

Each framework integration has unique features and considerations. Refer to the specific framework documentation for detailed implementation guidance:

- See individual framework files in this directory for comprehensive integration guides
- Check the official BentoML documentation for the latest API updates
- Join the BentoML community for framework-specific discussions and support

---

*Framework support is continuously expanding. Check the official documentation for the most up-to-date list of supported frameworks and their capabilities.*