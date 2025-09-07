# BentoML Transformers Integration

Comprehensive API reference for integrating Hugging Face Transformers models with BentoML.

## Overview

BentoML provides seamless integration with Hugging Face Transformers, supporting pipelines, custom models, tokenizers, and various NLP/computer vision tasks.

## Core Functions

### save_model()

Save a Transformers model to BentoML's model store.

```python
bentoml.transformers.save_model(
    name: str,
    model: Union[transformers.Pipeline, transformers.PreTrainedModel],
    signatures: dict = None,
    labels: dict = None,
    custom_objects: dict = None,
    metadata: dict = None
) -> bentoml.Tag
```

**Parameters:**
- `name` (str): Model name for identification
- `model` (Union[Pipeline, PreTrainedModel]): Transformers model or pipeline
- `signatures` (dict, optional): Model signature configuration
- `labels` (dict, optional): Labels for model organization
- `custom_objects` (dict, optional): Tokenizers, processors, etc.
- `metadata` (dict, optional): Custom metadata

**Returns:** `bentoml.Tag` with model name and version

#### Basic Pipeline Usage

```python
import bentoml
from transformers import pipeline

# Create pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Save pipeline
tag = bentoml.transformers.save_model("sentiment_analyzer", sentiment_pipeline)
print(f"Model saved: {tag}")
```

#### Custom Model with Tokenizer

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save with custom objects
tag = bentoml.transformers.save_model(
    name="bert_encoder",
    model=model,
    custom_objects={
        "tokenizer": tokenizer
    },
    metadata={
        "model_name": model_name,
        "task": "feature_extraction",
        "max_length": 512
    }
)
```

#### Advanced Configuration

```python
from transformers import pipeline, AutoTokenizer

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    device=0 if torch.cuda.is_available() else -1
)

# Save with comprehensive configuration
tag = bentoml.transformers.save_model(
    name="text_generator",
    model=generator,
    signatures={
        "generate": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "task": "text_generation",
        "model_family": "gpt",
        "stage": "production"
    },
    metadata={
        "max_length": 100,
        "temperature": 0.8,
        "top_p": 0.9,
        "model_size": "117M parameters"
    },
    custom_objects={
        "generation_config": {
            "max_length": 100,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9
        }
    }
)
```

### load_model()

Load a saved Transformers model from BentoML's model store.

```python
bentoml.transformers.load_model(
    tag: str
) -> Union[transformers.Pipeline, transformers.PreTrainedModel]
```

**Parameters:**
- `tag` (str): Model tag (name:version or name for latest)

**Returns:** Transformers pipeline or model

#### Basic Loading

```python
# Load latest version
pipeline = bentoml.transformers.load_model("sentiment_analyzer:latest")

# Load specific version
pipeline = bentoml.transformers.load_model("sentiment_analyzer:v1.2.0")

# Use loaded pipeline
result = pipeline("I love this product!")
print(result)
```

## Service Integration

### Text Classification Service

```python
import bentoml
from typing import List, Union

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
class TextClassificationService:
    def __init__(self):
        # Load sentiment analysis pipeline
        self.sentiment_pipeline = bentoml.transformers.load_model("sentiment_analyzer:latest")
        
        # Load custom objects if available
        model_ref = bentoml.transformers.get("sentiment_analyzer:latest")
        self.labels_map = model_ref.custom_objects.get("labels_map", {})
    
    @bentoml.api
    def analyze_sentiment(self, text: str) -> dict:
        # Single text analysis
        result = self.sentiment_pipeline(text)
        
        return {
            "text": text,
            "label": result[0]["label"],
            "score": result[0]["score"],
            "confidence": result[0]["score"]
        }
    
    @bentoml.api
    def analyze_batch(self, texts: List[str]) -> List[dict]:
        # Batch processing
        results = self.sentiment_pipeline(texts)
        
        return [
            {
                "text": text,
                "label": result["label"],
                "score": result["score"]
            }
            for text, result in zip(texts, results)
        ]
```

### Text Generation Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class TextGenerationService:
    def __init__(self):
        self.generator = bentoml.transformers.load_model("text_generator:latest")
        
        # Load generation configuration
        model_ref = bentoml.transformers.get("text_generator:latest")
        self.generation_config = model_ref.custom_objects.get("generation_config", {})
    
    @bentoml.api
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> dict:
        # Use provided parameters or defaults
        config = self.generation_config.copy()
        if max_length is not None:
            config["max_length"] = max_length
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        
        # Generate text
        result = self.generator(
            prompt,
            **config,
            return_full_text=False,
            num_return_sequences=1
        )
        
        return {
            "prompt": prompt,
            "generated_text": result[0]["generated_text"],
            "full_text": prompt + result[0]["generated_text"],
            "generation_config": config
        }
    
    @bentoml.api
    def generate_multiple(
        self, 
        prompt: str, 
        num_sequences: int = 3,
        max_length: int = 50
    ) -> dict:
        # Generate multiple sequences
        results = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            return_full_text=False,
            do_sample=True,
            temperature=0.8
        )
        
        return {
            "prompt": prompt,
            "generated_sequences": [r["generated_text"] for r in results],
            "num_sequences": len(results)
        }
```

### Question Answering Service

```python
@bentoml.service
class QuestionAnsweringService:
    def __init__(self):
        # Load QA pipeline
        self.qa_pipeline = bentoml.transformers.load_model("qa_model:latest")
        
        # Load model metadata
        model_ref = bentoml.transformers.get("qa_model:latest")
        self.max_answer_length = model_ref.metadata.get("max_answer_length", 15)
        self.max_seq_length = model_ref.metadata.get("max_seq_length", 384)
    
    @bentoml.api
    def answer_question(self, question: str, context: str) -> dict:
        # Get answer from context
        result = self.qa_pipeline({
            "question": question,
            "context": context
        })
        
        return {
            "question": question,
            "answer": result["answer"],
            "confidence": result["score"],
            "start_position": result.get("start", -1),
            "end_position": result.get("end", -1)
        }
    
    @bentoml.api
    def answer_multiple_questions(
        self, 
        questions: List[str], 
        context: str
    ) -> List[dict]:
        # Answer multiple questions about the same context
        results = []
        
        for question in questions:
            result = self.qa_pipeline({
                "question": question,
                "context": context
            })
            
            results.append({
                "question": question,
                "answer": result["answer"],
                "confidence": result["score"]
            })
        
        return results
```

### Named Entity Recognition Service

```python
@bentoml.service
class NERService:
    def __init__(self):
        self.ner_pipeline = bentoml.transformers.load_model("ner_model:latest")
        
        # Load entity mapping
        model_ref = bentoml.transformers.get("ner_model:latest")
        self.entity_labels = model_ref.custom_objects.get("entity_labels", {})
    
    @bentoml.api
    def extract_entities(self, text: str) -> dict:
        # Extract named entities
        entities = self.ner_pipeline(text)
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity["entity_group"] if "entity_group" in entity else entity["entity"]
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            
            grouped_entities[entity_type].append({
                "text": entity["word"],
                "confidence": entity["score"],
                "start": entity.get("start"),
                "end": entity.get("end")
            })
        
        return {
            "text": text,
            "entities": entities,
            "grouped_entities": grouped_entities,
            "entity_count": len(entities)
        }
```

### Multi-Task NLP Service

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "12Gi"}
)
class MultiTaskNLPService:
    def __init__(self):
        # Load multiple NLP pipelines
        self.sentiment = bentoml.transformers.load_model("sentiment:latest")
        self.ner = bentoml.transformers.load_model("ner:latest")
        self.summarizer = bentoml.transformers.load_model("summarizer:latest")
        self.qa = bentoml.transformers.load_model("qa:latest")
    
    @bentoml.api
    def analyze_text_comprehensive(self, text: str) -> dict:
        # Perform multiple NLP tasks
        analysis = {}
        
        try:
            # Sentiment analysis
            sentiment = self.sentiment(text)
            analysis["sentiment"] = {
                "label": sentiment[0]["label"],
                "confidence": sentiment[0]["score"]
            }
        except Exception as e:
            analysis["sentiment"] = {"error": str(e)}
        
        try:
            # Named entity recognition
            entities = self.ner(text)
            analysis["entities"] = entities
        except Exception as e:
            analysis["entities"] = {"error": str(e)}
        
        try:
            # Text summarization (if text is long enough)
            if len(text.split()) > 50:
                summary = self.summarizer(text, max_length=50, min_length=10)
                analysis["summary"] = summary[0]["summary_text"]
            else:
                analysis["summary"] = "Text too short for summarization"
        except Exception as e:
            analysis["summary"] = {"error": str(e)}
        
        return {
            "original_text": text,
            "analysis": analysis,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    @bentoml.api
    def question_answering(self, question: str, context: str) -> dict:
        try:
            result = self.qa({
                "question": question,
                "context": context
            })
            return {
                "question": question,
                "answer": result["answer"],
                "confidence": result["score"]
            }
        except Exception as e:
            return {"error": str(e)}
```

## Vision-Language Models

### Image Captioning Service

```python
from PIL import Image
import bentoml

@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class ImageCaptioningService:
    def __init__(self):
        # Load image-to-text pipeline
        self.captioning_pipeline = bentoml.transformers.load_model("image_captioner:latest")
        
        # Load generation parameters
        model_ref = bentoml.transformers.get("image_captioner:latest")
        self.generation_config = model_ref.custom_objects.get("generation_config", {})
    
    @bentoml.api
    def generate_caption(self, image: Image.Image) -> dict:
        # Generate caption for image
        result = self.captioning_pipeline(image, **self.generation_config)
        
        return {
            "caption": result[0]["generated_text"],
            "confidence": result[0].get("score", 1.0),
            "image_size": image.size
        }
    
    @bentoml.api
    def generate_multiple_captions(
        self, 
        image: Image.Image, 
        num_captions: int = 3
    ) -> dict:
        # Generate multiple captions
        results = self.captioning_pipeline(
            image,
            num_return_sequences=num_captions,
            do_sample=True,
            temperature=0.8
        )
        
        captions = [r["generated_text"] for r in results]
        
        return {
            "captions": captions,
            "num_captions": len(captions),
            "image_size": image.size
        }
```

## Advanced Features

### Custom Pipeline Creation

```python
from transformers import AutoModel, AutoTokenizer, Pipeline

class CustomClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}
    
    def preprocess(self, inputs):
        return self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, **kwargs):
        logits = model_outputs.logits
        probs = torch.softmax(logits, dim=-1)
        return [{"label": "POSITIVE" if p > 0.5 else "NEGATIVE", "score": float(p)} 
                for p in probs[:, 1]]

# Create and save custom pipeline
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

custom_pipeline = CustomClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    task="custom-classification"
)

tag = bentoml.transformers.save_model("custom_classifier", custom_pipeline)
```

### Model Fine-tuning Integration

```python
@bentoml.service
class FineTunableService:
    def __init__(self):
        self.model = bentoml.transformers.load_model("finetune_base:latest")
        
        # Load model reference for custom objects
        model_ref = bentoml.transformers.get("finetune_base:latest")
        self.tokenizer = model_ref.custom_objects["tokenizer"]
    
    @bentoml.api
    def predict(self, text: str) -> dict:
        # Standard prediction
        result = self.model(text)
        return {"prediction": result[0]["label"], "confidence": result[0]["score"]}
    
    @bentoml.api
    def update_model(self, examples: List[dict]) -> dict:
        # Simple example of online learning/adaptation
        # In practice, this would involve more sophisticated fine-tuning
        
        training_texts = [ex["text"] for ex in examples]
        training_labels = [ex["label"] for ex in examples]
        
        # Placeholder for fine-tuning logic
        # This would typically involve:
        # 1. Tokenizing new examples
        # 2. Computing gradients
        # 3. Updating model parameters
        
        return {
            "message": f"Model updated with {len(examples)} examples",
            "examples_processed": len(examples)
        }
```

## Performance Optimization

### Batch Processing

```python
@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class OptimizedTransformersService:
    def __init__(self):
        self.pipeline = bentoml.transformers.load_model("optimized_model:latest")
        self.batch_size = 16
    
    @bentoml.api
    def process_batch(self, texts: List[str]) -> List[dict]:
        # Process in optimal batch sizes
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
        
        return [
            {
                "text": text,
                "result": result
            }
            for text, result in zip(texts, results)
        ]
```

### Model Quantization

```python
# Save quantized model
from transformers import pipeline
import torch

# Create pipeline with quantization
pipeline_quantized = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0,
    torch_dtype=torch.float16  # Use half precision
)

tag = bentoml.transformers.save_model(
    "quantized_sentiment",
    pipeline_quantized,
    metadata={"quantization": "fp16", "memory_optimized": True}
)
```

## Best Practices

### 1. Error Handling

```python
@bentoml.service
class RobustTransformersService:
    def __init__(self):
        try:
            self.pipeline = bentoml.transformers.load_model("robust_model:latest")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    @bentoml.api
    def predict(self, text: str) -> dict:
        if not self.model_loaded:
            return {"error": "Model not available"}
        
        try:
            # Validate input
            if not text or len(text.strip()) == 0:
                return {"error": "Input text cannot be empty"}
            
            if len(text) > 5000:  # Token limit check
                return {"error": "Input text too long (max 5000 characters)"}
            
            result = self.pipeline(text)
            
            return {
                "success": True,
                "result": result[0] if isinstance(result, list) else result
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Prediction failed", "details": str(e)}
```

### 2. Memory Management

```python
@bentoml.service(resources={"gpu": "1", "memory": "12Gi"})
class MemoryOptimizedService:
    def __init__(self):
        self.pipeline = bentoml.transformers.load_model("memory_opt:latest")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.pipeline.model, 'gradient_checkpointing_enable'):
            self.pipeline.model.gradient_checkpointing_enable()
    
    @bentoml.api
    def predict(self, text: str) -> dict:
        with torch.no_grad():  # Disable gradients for inference
            result = self.pipeline(text)
            
            # Clear cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {"result": result[0]}
```

### 3. Model Versioning

```python
# Save with comprehensive versioning
tag = bentoml.transformers.save_model(
    "production_model",
    pipeline,
    labels={
        "version": "2.1.0",
        "environment": "production",
        "model_family": "roberta"
    },
    metadata={
        "transformers_version": transformers.__version__,
        "base_model": "roberta-base",
        "fine_tuned_on": "custom_dataset_v2",
        "accuracy": 0.94,
        "f1_score": 0.93,
        "created_at": datetime.now().isoformat()
    }
)
```

For more Transformers examples, visit the [BentoML examples repository](https://github.com/bentoml/BentoML/tree/main/examples/) and the [Hugging Face documentation](https://huggingface.co/docs/transformers/).