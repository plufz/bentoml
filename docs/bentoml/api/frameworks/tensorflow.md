# BentoML TensorFlow Integration

Comprehensive API reference for integrating TensorFlow models with BentoML.

## Overview

BentoML provides robust integration with TensorFlow, supporting Keras models, native TensorFlow modules, SavedModel format, and various TensorFlow-specific features like RaggedTensors.

## Core Functions

### save_model()

Save a TensorFlow model to BentoML's model store.

```python
bentoml.tensorflow.save_model(
    name: str,
    model: tf.Module,
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    external_modules: list = None,
    metadata: dict = None
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification
- `model` (tf.Module): TensorFlow model or Keras model
- `signatures` (dict, optional): Model signature configuration
- `labels` (dict, optional): Labels for model organization  
- `custom_objects` (dict, optional): Additional objects to save
- `external_modules` (list, optional): External modules required
- `metadata` (dict, optional): Custom metadata

**Returns:** `bentoml.Tag` with model name and version

#### Basic Keras Model

```python
import tensorflow as tf
import bentoml

# Define Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model (example with dummy data)
import numpy as np
X_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 10, (1000,))
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Save model
tag = bentoml.tensorflow.save_model("keras_classifier", model)
print(f"Model saved: {tag}")
```

#### Native TensorFlow Module

```python
import tensorflow as tf
import bentoml

class NativeModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.Variable(tf.random.normal([784, 128]))
        self.dense2 = tf.Variable(tf.random.normal([128, 10]))
        self.bias1 = tf.Variable(tf.zeros([128]))
        self.bias2 = tf.Variable(tf.zeros([10]))
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def __call__(self, x):
        hidden = tf.nn.relu(tf.matmul(x, self.dense1) + self.bias1)
        output = tf.matmul(hidden, self.dense2) + self.bias2
        return tf.nn.softmax(output)

# Create and save model
native_model = NativeModel()
tag = bentoml.tensorflow.save_model("native_classifier", native_model)
```

#### Advanced Configuration

```python
# Save with comprehensive configuration
tag = bentoml.tensorflow.save_model(
    name="production_tf_model",
    model=trained_model,
    signatures={
        "serving_default": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "stage": "production",
        "framework": "tensorflow",
        "version": "2.0.0"
    },
    metadata={
        "accuracy": 0.95,
        "loss": 0.12,
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "adam",
        "tensorflow_version": tf.__version__
    },
    custom_objects={
        "preprocessing_fn": preprocessing_function,
        "class_names": ["cat", "dog", "bird"]
    }
)
```

### load_model()

Load a saved TensorFlow model from BentoML's model store.

```python
bentoml.tensorflow.load_model(
    tag: str,
    device: str = "/CPU:0"
) -> tf.Module
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)
- `device` (str, optional): Device specification (default: "/CPU:0")

**Returns:** TensorFlow model or Keras model

#### Basic Loading

```python
# Load latest version
model = bentoml.tensorflow.load_model("keras_classifier:latest")

# Load specific version  
model = bentoml.tensorflow.load_model("keras_classifier:v1.2.0")

# Load to GPU if available
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
model = bentoml.tensorflow.load_model("keras_classifier:latest", device=device)
```

#### Device Management

```python
import tensorflow as tf

# Check available devices
physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)

# Load model to specific device
if tf.config.list_physical_devices('GPU'):
    model = bentoml.tensorflow.load_model("model:latest", device="/GPU:0")
else:
    model = bentoml.tensorflow.load_model("model:latest", device="/CPU:0")
```

### get()

Get a BentoML model reference without loading into memory.

```python
model_ref = bentoml.tensorflow.get("model_name:latest")
# Access model path, metadata, custom objects
```

## Service Integration

### Keras Classification Service

```python
import bentoml
import tensorflow as tf
import numpy as np
from typing import List

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
class TensorFlowClassifier:
    def __init__(self):
        # Load model during service initialization
        self.model = bentoml.tensorflow.load_model("keras_classifier:latest")
        
        # Load custom objects
        model_ref = bentoml.tensorflow.get("keras_classifier:latest")
        self.class_names = model_ref.custom_objects.get("class_names", [])
        self.preprocessing_fn = model_ref.custom_objects.get("preprocessing_fn")
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        # Convert to tensor
        X = tf.constant(input_data, dtype=tf.float32)
        
        # Apply preprocessing if available
        if self.preprocessing_fn:
            X = self.preprocessing_fn(X)
        
        # Get predictions
        predictions = self.model(X)
        predicted_classes = tf.argmax(predictions, axis=1)
        confidences = tf.reduce_max(predictions, axis=1)
        
        return {
            "predictions": predicted_classes.numpy().tolist(),
            "class_names": [self.class_names[i] for i in predicted_classes.numpy()] if self.class_names else None,
            "confidences": confidences.numpy().tolist(),
            "probabilities": predictions.numpy().tolist()
        }
    
    @bentoml.api
    def predict_single(self, features: List[float]) -> dict:
        # Single prediction
        X = tf.constant([features], dtype=tf.float32)
        
        if self.preprocessing_fn:
            X = self.preprocessing_fn(X)
        
        prediction = self.model(X)[0]
        predicted_class = tf.argmax(prediction)
        confidence = tf.reduce_max(prediction)
        
        return {
            "prediction": int(predicted_class.numpy()),
            "class_name": self.class_names[predicted_class.numpy()] if self.class_names else None,
            "confidence": float(confidence.numpy()),
            "probabilities": prediction.numpy().tolist()
        }
```

### Image Classification Service

```python
import tensorflow as tf
from PIL import Image
import numpy as np

@bentoml.service(
    resources={"gpu": "1", "memory": "6Gi"}
)
class TensorFlowImageClassifier:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("image_classifier:latest")
        
        # Load model metadata
        model_ref = bentoml.tensorflow.get("image_classifier:latest")
        self.input_shape = model_ref.metadata.get("input_shape", (224, 224, 3))
        self.class_names = model_ref.custom_objects.get("class_names", [])
    
    def preprocess_image(self, image: Image.Image) -> tf.Tensor:
        # Resize and normalize image
        image = image.convert('RGB')
        image = image.resize(self.input_shape[:2])
        
        # Convert to tensor and normalize
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
        image_array = tf.keras.applications.imagenet_utils.preprocess_input(image_array)
        
        return image_array
    
    @bentoml.api
    def classify_image(self, image: Image.Image) -> dict:
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        predictions = self.model(processed_image)
        predicted_class = tf.argmax(predictions[0])
        confidence = tf.reduce_max(predictions[0])
        
        # Get top 5 predictions
        top5_indices = tf.nn.top_k(predictions[0], k=5).indices
        top5_scores = tf.nn.top_k(predictions[0], k=5).values
        
        top5_predictions = []
        for i, (idx, score) in enumerate(zip(top5_indices, top5_scores)):
            top5_predictions.append({
                "class_id": int(idx.numpy()),
                "class_name": self.class_names[idx.numpy()] if self.class_names else f"class_{idx.numpy()}",
                "confidence": float(score.numpy())
            })
        
        return {
            "predicted_class": int(predicted_class.numpy()),
            "predicted_class_name": self.class_names[predicted_class.numpy()] if self.class_names else None,
            "confidence": float(confidence.numpy()),
            "top5_predictions": top5_predictions
        }
```

### Text Processing Service

```python
@bentoml.service
class TensorFlowTextProcessor:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("text_classifier:latest")
        
        # Load tokenizer and vocabulary
        model_ref = bentoml.tensorflow.get("text_classifier:latest")
        self.tokenizer = model_ref.custom_objects.get("tokenizer")
        self.max_length = model_ref.metadata.get("max_length", 100)
        self.vocab_size = model_ref.metadata.get("vocab_size", 10000)
    
    def preprocess_text(self, text: str) -> tf.Tensor:
        # Tokenize text
        if self.tokenizer:
            tokens = self.tokenizer.texts_to_sequences([text])
        else:
            # Simple tokenization fallback
            tokens = [[hash(word) % self.vocab_size for word in text.split()]]
        
        # Pad sequences
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            tokens, maxlen=self.max_length, truncating='post'
        )
        
        return tf.constant(padded, dtype=tf.int32)
    
    @bentoml.api
    def classify_text(self, text: str) -> dict:
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get prediction
        predictions = self.model(processed_text)
        predicted_class = tf.argmax(predictions[0])
        confidence = tf.reduce_max(predictions[0])
        
        return {
            "text": text,
            "predicted_class": int(predicted_class.numpy()),
            "confidence": float(confidence.numpy()),
            "probabilities": predictions[0].numpy().tolist()
        }
```

## Advanced Features

### SavedModel with Signatures

```python
# Save model with custom signatures
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def predict_fn(x):
    return model(x)

# Save with signature
concrete_function = predict_fn.get_concrete_function()
tag = bentoml.tensorflow.save_model(
    "signed_model",
    model,
    signatures={"predict": concrete_function}
)
```

### RaggedTensor Support

```python
import tensorflow as tf

class RaggedModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.embedding = tf.Variable(tf.random.normal([1000, 64]))
    
    @tf.function(
        input_signature=[tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32)]
    )
    def __call__(self, ragged_input):
        embedded = tf.nn.embedding_lookup(self.embedding, ragged_input)
        return tf.reduce_mean(embedded, axis=1)

# Save RaggedTensor model
ragged_model = RaggedModel()
tag = bentoml.tensorflow.save_model("ragged_model", ragged_model)

# Service with RaggedTensor support
@bentoml.service
class RaggedTensorService:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("ragged_model:latest")
    
    @bentoml.api
    def process_sequences(self, sequences: List[List[int]]) -> dict:
        # Create RaggedTensor
        ragged_input = tf.ragged.constant(sequences, dtype=tf.int32)
        
        # Process with model
        result = self.model(ragged_input)
        
        return {
            "embeddings": result.numpy().tolist(),
            "input_shapes": [len(seq) for seq in sequences]
        }
```

### Custom Training Loop Integration

```python
class TrainableService(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.dense(x)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.dense.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_variables))
        
        return loss
    
    @tf.function
    def predict(self, x):
        return self.dense(x)

@bentoml.service
class OnlineTrainingService:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("trainable_model:latest")
    
    @bentoml.api
    def predict(self, features: List[float]) -> dict:
        X = tf.constant([features], dtype=tf.float32)
        prediction = self.model.predict(X)
        
        return {
            "prediction": tf.argmax(prediction[0]).numpy().item(),
            "probabilities": prediction[0].numpy().tolist()
        }
    
    @bentoml.api
    def update_model(self, features: List[float], label: int) -> dict:
        X = tf.constant([features], dtype=tf.float32)
        y = tf.constant([label], dtype=tf.int32)
        
        loss = self.model.train_step(X, y)
        
        return {
            "training_loss": float(loss.numpy()),
            "message": "Model updated successfully"
        }
```

## Performance Optimization

### Mixed Precision

```python
# Enable mixed precision for faster training/inference
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

@bentoml.service(resources={"gpu": "1"})
class MixedPrecisionService:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("fp16_model:latest")
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        X = tf.constant(input_data, dtype=tf.float32)
        
        # Model automatically uses mixed precision
        predictions = self.model(X)
        
        return {
            "predictions": predictions.numpy().tolist()
        }
```

### Graph Optimization

```python
@bentoml.service
class OptimizedService:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("optimized_model:latest")
        
        # Create optimized graph
        self._predict_fn = tf.function(self.model.__call__)
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        X = tf.constant(input_data, dtype=tf.float32)
        
        # Use optimized function
        predictions = self._predict_fn(X)
        
        return {
            "predictions": predictions.numpy().tolist()
        }
```

### Batch Processing

```python
@bentoml.service
class BatchTensorFlowService:
    def __init__(self):
        self.model = bentoml.tensorflow.load_model("batch_model:latest")
        self.batch_size = 32
    
    @bentoml.api
    def batch_predict(self, input_batch: List[List[float]]) -> List[dict]:
        # Process in batches for memory efficiency
        results = []
        
        for i in range(0, len(input_batch), self.batch_size):
            batch = input_batch[i:i + self.batch_size]
            X = tf.constant(batch, dtype=tf.float32)
            
            predictions = self.model(X)
            
            # Convert batch results to individual predictions
            for j, pred in enumerate(predictions):
                results.append({
                    "input_index": i + j,
                    "prediction": tf.argmax(pred).numpy().item(),
                    "confidence": tf.reduce_max(pred).numpy().item()
                })
        
        return results
```

## Best Practices

### 1. Model Versioning and Metadata

```python
# Save comprehensive model information
tag = bentoml.tensorflow.save_model(
    "versioned_model",
    model,
    metadata={
        "tensorflow_version": tf.__version__,
        "model_architecture": "CNN",
        "training_dataset": "CIFAR-10",
        "accuracy": 0.92,
        "loss": 0.25,
        "epochs": 50,
        "learning_rate": 0.001,
        "created_at": datetime.now().isoformat()
    },
    labels={
        "environment": "production",
        "team": "ml-team",
        "use_case": "image_classification"
    }
)
```

### 2. Error Handling

```python
@bentoml.service
class RobustTensorFlowService:
    def __init__(self):
        try:
            self.model = bentoml.tensorflow.load_model("robust_model:latest")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        if not self.model_loaded:
            return {"error": "Model not available"}
        
        try:
            X = tf.constant(input_data, dtype=tf.float32)
            
            # Validate input shape
            if len(X.shape) != 2:
                return {"error": f"Expected 2D input, got {len(X.shape)}D"}
            
            predictions = self.model(X)
            
            return {
                "success": True,
                "predictions": predictions.numpy().tolist()
            }
            
        except tf.errors.InvalidArgumentError as e:
            return {"error": f"Invalid input: {str(e)}"}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Prediction failed"}
```

### 3. Resource Management

```python
@bentoml.service(resources={"gpu": "1", "memory": "8Gi"})
class ResourceManagedService:
    def __init__(self):
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        
        self.model = bentoml.tensorflow.load_model("gpu_model:latest")
    
    @bentoml.api
    def predict(self, input_data: List[List[float]]) -> dict:
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            X = tf.constant(input_data, dtype=tf.float32)
            predictions = self.model(X)
            
            return {
                "predictions": predictions.numpy().tolist(),
                "device_used": "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
            }
```

For more TensorFlow examples, visit the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/).