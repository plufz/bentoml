# BentoML Model Composition

Guide to combining multiple AI models to build sophisticated applications using BentoML's composition patterns.

## Overview

Model composition in BentoML allows you to:
- **Combine multiple AI models** to build sophisticated applications
- **Support sequential and parallel workflows** for complex processing
- **Enable processing different data types** with specialized models
- **Improve accuracy** through ensemble methods
- **Run models on specialized hardware** with independent scaling

## Composition Patterns

### 1. Single Service Model Composition

Run multiple models within the same service on shared hardware:

```python
import bentoml
from transformers import pipeline

@bentoml.service(
    resources={"gpu": "1"}
)
class MultiModelService:
    def __init__(self):
        # Load multiple models in the same service
        self.classifier = pipeline("zero-shot-classification")
        self.sentiment = pipeline("sentiment-analysis")
        self.generator = pipeline("text-generation")
    
    @bentoml.api
    def classify_and_analyze(self, text: str, labels: list[str]) -> dict:
        # Use multiple models in sequence
        classification = self.classifier(text, labels)
        sentiment = self.sentiment(text)
        
        return {
            "classification": classification,
            "sentiment": sentiment
        }
    
    @bentoml.api
    def generate_and_score(self, prompt: str) -> dict:
        # Generate text and analyze it
        generated = self.generator(prompt, max_length=100)
        text = generated[0]["generated_text"]
        sentiment = self.sentiment(text)
        
        return {
            "generated_text": text,
            "sentiment_score": sentiment
        }
```

**Benefits:**
- Shared hardware resources
- Lower latency (no network calls)
- Simpler deployment

### 2. Multi-Service Model Composition

Create independent services with separate scaling and hardware requirements:

#### Sequential Processing

```python
# Service 1: Text Generator
@bentoml.service(
    resources={"gpu": "1", "memory": "8Gi"}
)
class TextGenerator:
    def __init__(self):
        self.model = pipeline("text-generation", model="gpt2-large")
    
    @bentoml.api
    def generate(self, prompt: str, max_length: int = 100) -> str:
        result = self.model(prompt, max_length=max_length)
        return result[0]["generated_text"]

# Service 2: Text Classifier  
@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"}
)
class TextClassifier:
    def __init__(self):
        self.model = pipeline("text-classification")
    
    @bentoml.api
    def classify(self, text: str) -> dict:
        return self.model(text)[0]

# Composed Service
@bentoml.service
class ComposedTextService:
    generator = bentoml.depends(TextGenerator)
    classifier = bentoml.depends(TextClassifier)
    
    @bentoml.api
    def generate_and_classify(self, prompt: str) -> dict:
        # Sequential: generate then classify
        generated_text = self.generator.generate(prompt)
        classification = self.classifier.classify(generated_text)
        
        return {
            "generated_text": generated_text,
            "classification": classification
        }
```

#### Concurrent Processing

```python
import asyncio

@bentoml.service
class ConcurrentService:
    model1 = bentoml.depends(TextGenerator)
    model2 = bentoml.depends(TextClassifier)
    
    @bentoml.api
    async def parallel_processing(self, text: str) -> dict:
        # Run both models concurrently
        results = await asyncio.gather(
            self.model1.generate.to_async()(text),
            self.model2.classify.to_async()(text)
        )
        
        return {
            "generated": results[0],
            "classified": results[1]
        }
```

### 3. Inference Graph (Complex Workflows)

Build sophisticated AI pipelines with parallel and sequential processing:

```python
@bentoml.service
class InferenceGraphService:
    generator1 = bentoml.depends(TextGenerator)
    generator2 = bentoml.depends(TextGenerator) 
    classifier = bentoml.depends(TextClassifier)
    
    @bentoml.api
    async def complex_workflow(self, prompt: str) -> dict:
        # Step 1: Generate text using multiple models in parallel
        generated_texts = await asyncio.gather(
            self.generator1.generate.to_async()(prompt),
            self.generator2.generate.to_async()(prompt + " Alternative:")
        )
        
        # Step 2: Score all generated texts using classifier
        scores = await asyncio.gather(*[
            self.classifier.classify.to_async()(text) 
            for text in generated_texts
        ])
        
        # Step 3: Combine results
        results = []
        for text, score in zip(generated_texts, scores):
            results.append({
                "text": text,
                "score": score["score"],
                "label": score["label"]
            })
        
        # Return best result
        best_result = max(results, key=lambda x: x["score"])
        
        return {
            "best_generation": best_result,
            "all_results": results
        }
```

## Key Technical Features

### Service Dependencies

Use `bentoml.depends()` to inject other services:

```python
@bentoml.service
class ParentService:
    child_service = bentoml.depends(ChildService)
    
    @bentoml.api
    def use_child(self, data: str) -> dict:
        result = self.child_service.process(data)
        return {"processed": result}
```

### Async Conversion

Convert synchronous service methods to async for concurrent execution:

```python
# Synchronous call
result = self.service.method(data)

# Asynchronous call
result = await self.service.method.to_async()(data)
```

### Resource Configuration

Configure different resources for different services:

```python
@bentoml.service(
    resources={
        "gpu": "1",           # For GPU-intensive models
        "memory": "16Gi",     # High memory for large models
        "cpu": "4"            # CPU cores
    }
)
class GPUService:
    pass

@bentoml.service(
    resources={
        "cpu": "2",           # CPU-only service
        "memory": "4Gi"
    }
)
class CPUService:
    pass
```

## Use Cases and Examples

### Multi-Modal Processing

```python
@bentoml.service
class MultiModalService:
    image_model = bentoml.depends(ImageClassifier)
    text_model = bentoml.depends(TextAnalyzer)
    
    @bentoml.api
    async def analyze_image_and_caption(
        self, 
        image: bytes, 
        caption: str
    ) -> dict:
        # Process image and text concurrently
        image_result, text_result = await asyncio.gather(
            self.image_model.classify.to_async()(image),
            self.text_model.analyze.to_async()(caption)
        )
        
        return {
            "image_classification": image_result,
            "text_analysis": text_result,
            "combined_confidence": (
                image_result["confidence"] * text_result["confidence"]
            ) ** 0.5
        }
```

### Ensemble Predictions

```python
@bentoml.service  
class EnsembleService:
    model1 = bentoml.depends(ClassifierA)
    model2 = bentoml.depends(ClassifierB)
    model3 = bentoml.depends(ClassifierC)
    
    @bentoml.api
    async def ensemble_predict(self, data: str) -> dict:
        # Get predictions from all models
        predictions = await asyncio.gather(
            self.model1.predict.to_async()(data),
            self.model2.predict.to_async()(data),
            self.model3.predict.to_async()(data)
        )
        
        # Aggregate results (voting/averaging)
        confidence_scores = [p["confidence"] for p in predictions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Majority voting for label
        labels = [p["label"] for p in predictions]
        final_label = max(set(labels), key=labels.count)
        
        return {
            "ensemble_prediction": final_label,
            "confidence": avg_confidence,
            "individual_predictions": predictions
        }
```

### Pipeline Processing

```python
@bentoml.service
class DataPipeline:
    preprocessor = bentoml.depends(DataPreprocessor)
    model = bentoml.depends(MLModel) 
    postprocessor = bentoml.depends(DataPostprocessor)
    
    @bentoml.api
    def process_pipeline(self, raw_data: dict) -> dict:
        # Sequential pipeline processing
        cleaned_data = self.preprocessor.clean(raw_data)
        prediction = self.model.predict(cleaned_data)
        final_result = self.postprocessor.format(prediction)
        
        return {
            "input": raw_data,
            "processed": final_result,
            "pipeline_steps": ["preprocess", "predict", "postprocess"]
        }
```

## Best Practices

1. **Choose the Right Pattern**:
   - Single service for tightly coupled models
   - Multi-service for independent scaling needs
   - Complex graphs for sophisticated workflows

2. **Resource Optimization**:
   - Assign appropriate resources to each service
   - Use GPU services for compute-intensive models
   - Use CPU services for lightweight processing

3. **Error Handling**:
   ```python
   @bentoml.api
   async def robust_composition(self, data: str) -> dict:
       try:
           result1 = await self.service1.process.to_async()(data)
       except Exception as e:
           result1 = {"error": str(e), "fallback": True}
       
       try:
           result2 = await self.service2.process.to_async()(data)
       except Exception as e:
           result2 = {"error": str(e), "fallback": True}
       
       return {"service1": result1, "service2": result2}
   ```

4. **Monitoring and Logging**:
   ```python
   import logging
   
   @bentoml.service
   class MonitoredComposition:
       def __init__(self):
           self.logger = logging.getLogger(__name__)
       
       @bentoml.api
       async def monitored_process(self, data: str) -> dict:
           self.logger.info(f"Starting composition for data length: {len(data)}")
           
           start_time = time.time()
           result = await self.complex_processing(data)
           processing_time = time.time() - start_time
           
           self.logger.info(f"Composition completed in {processing_time:.2f}s")
           
           return {**result, "processing_time": processing_time}
   ```

Model composition in BentoML provides powerful patterns for building sophisticated AI applications that can scale independently and handle complex workflows efficiently.