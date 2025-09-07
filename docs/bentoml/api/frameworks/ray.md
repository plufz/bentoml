# BentoML Ray Integration

Comprehensive guide to integrating Ray distributed computing framework with BentoML for scalable model serving and distributed workloads.

## Overview

BentoML provides integration with Ray, a distributed computing framework, enabling:
- **Distributed model serving** with Ray Serve
- **Scalable batch processing** across multiple nodes
- **Elastic scaling** based on demand
- **Resource management** for compute-intensive workloads
- **Multi-node deployment** for high-throughput scenarios

*Note: For detailed Ray integration, refer to the BentoRay guide in the official documentation.*

## Core Concepts

### Ray Serve Integration

Ray Serve is Ray's model serving library that can be integrated with BentoML for distributed deployment.

```python
import bentoml
import ray
from ray import serve

# Initialize Ray
ray.init()

@serve.deployment
@bentoml.service
class RayBentoMLService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("my_model:latest")
    
    @bentoml.api
    async def predict(self, input_data: list) -> dict:
        prediction = self.model.predict([input_data])
        return {"prediction": prediction[0]}

# Deploy with Ray Serve
RayBentoMLService.deploy()
```

### Distributed Model Loading

```python
import ray
import bentoml
from typing import List

@ray.remote
class DistributedModelWorker:
    def __init__(self, model_tag: str):
        self.model = bentoml.pytorch.load_model(model_tag)
    
    def predict(self, batch_data):
        return self.model.predict(batch_data)

@bentoml.service
class RayDistributedService:
    def __init__(self):
        # Initialize Ray workers
        self.workers = [
            DistributedModelWorker.remote("model:latest") 
            for _ in range(4)  # 4 workers
        ]
    
    @bentoml.api
    async def distributed_predict(self, input_batches: List[List]) -> List[dict]:
        # Distribute work across Ray workers
        futures = []
        for i, batch in enumerate(input_batches):
            worker = self.workers[i % len(self.workers)]
            future = worker.predict.remote(batch)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        return [{"batch_id": i, "predictions": result} for i, result in enumerate(results)]
```

## Advanced Integration Patterns

### Elastic Scaling with Ray Serve

```python
import bentoml
import ray
from ray import serve
from ray.serve.config import AutoscalingConfig

@serve.deployment(
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=10,
        target_num_ongoing_requests_per_replica=2,
    )
)
@bentoml.service
class ElasticService:
    def __init__(self):
        self.model = bentoml.transformers.load_model("bert_model:latest")
    
    @bentoml.api
    async def classify_text(self, text: str) -> dict:
        result = self.model(text)
        return {
            "text": text,
            "classification": result[0]["label"],
            "confidence": result[0]["score"]
        }

# Deploy with elastic scaling
app = ElasticService.bind()
serve.run(app, host="0.0.0.0", port=8000)
```

### Multi-GPU Distributed Inference

```python
import ray
import torch
import bentoml
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote(num_gpus=1)
class GPUModelWorker:
    def __init__(self, model_tag: str, device_id: int):
        # Load model on specific GPU
        self.device = f"cuda:{device_id}"
        self.model = bentoml.pytorch.load_model(model_tag, device_id=self.device)
    
    def predict(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            result = self.model(input_tensor)
            return result.cpu()

@bentoml.service
class MultiGPUService:
    def __init__(self):
        # Create placement group for GPU allocation
        self.pg = placement_group([{"GPU": 1} for _ in range(4)])
        ray.get(self.pg.ready())
        
        # Initialize GPU workers
        self.gpu_workers = [
            GPUModelWorker.remote("gpu_model:latest", i)
            for i in range(4)
        ]
    
    @bentoml.api
    async def parallel_inference(self, input_batches: List) -> List[dict]:
        # Distribute inference across GPUs
        futures = []
        for i, batch in enumerate(input_batches):
            worker = self.gpu_workers[i % len(self.gpu_workers)]
            future = worker.predict.remote(torch.tensor(batch))
            futures.append(future)
        
        results = ray.get(futures)
        
        return [
            {"gpu_id": i % len(self.gpu_workers), "result": result.tolist()}
            for i, result in enumerate(results)
        ]
```

### Distributed Batch Processing

```python
import ray
import bentoml
from typing import List, Iterator
import numpy as np

@ray.remote
class BatchProcessor:
    def __init__(self, model_tag: str):
        self.model = bentoml.sklearn.load_model(model_tag)
    
    def process_batch(self, batch_data: List) -> List:
        # Process batch of data
        predictions = self.model.predict(batch_data)
        return predictions.tolist()

@bentoml.service
class DistributedBatchService:
    def __init__(self):
        # Initialize distributed batch processors
        self.num_workers = 8
        self.processors = [
            BatchProcessor.remote("batch_model:latest")
            for _ in range(self.num_workers)
        ]
    
    def chunk_data(self, data: List, chunk_size: int) -> Iterator[List]:
        """Split data into chunks for distributed processing"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    @bentoml.api
    async def process_large_batch(
        self, 
        data: List[List[float]], 
        chunk_size: int = 100
    ) -> dict:
        # Split data into chunks
        chunks = list(self.chunk_data(data, chunk_size))
        
        # Distribute chunks across workers
        futures = []
        for i, chunk in enumerate(chunks):
            worker = self.processors[i % self.num_workers]
            future = worker.process_batch.remote(chunk)
            futures.append(future)
        
        # Collect results
        chunk_results = ray.get(futures)
        
        # Combine results
        all_predictions = []
        for chunk_result in chunk_results:
            all_predictions.extend(chunk_result)
        
        return {
            "total_samples": len(data),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "predictions": all_predictions,
            "workers_used": self.num_workers
        }
```

### Ray Dataset Integration

```python
import ray
import bentoml
from ray.data import Dataset

@bentoml.service
class RayDatasetService:
    def __init__(self):
        self.model = bentoml.transformers.load_model("text_classifier:latest")
    
    @ray.remote
    def predict_batch_udf(self, batch):
        """User-defined function for Ray Dataset processing"""
        texts = batch["text"]
        results = []
        
        for text in texts:
            result = self.model(text)
            results.append({
                "text": text,
                "prediction": result[0]["label"],
                "confidence": result[0]["score"]
            })
        
        return {"predictions": results}
    
    @bentoml.api
    async def process_dataset(self, texts: List[str]) -> dict:
        # Create Ray Dataset
        dataset = ray.data.from_items([{"text": text} for text in texts])
        
        # Process with distributed UDF
        results_dataset = dataset.map_batches(
            self.predict_batch_udf,
            batch_size=32,
            num_cpus=1
        )
        
        # Collect results
        results = results_dataset.take_all()
        
        # Flatten predictions
        all_predictions = []
        for batch_result in results:
            all_predictions.extend(batch_result["predictions"])
        
        return {
            "total_processed": len(texts),
            "predictions": all_predictions,
            "processing_method": "ray_dataset"
        }
```

## Performance Optimization

### Resource Management

```python
import ray
import bentoml
from ray.util.placement_group import placement_group

@bentoml.service
class ResourceOptimizedService:
    def __init__(self):
        # Configure Ray cluster resources
        self.cluster_resources = ray.cluster_resources()
        
        # Create placement group for guaranteed resources
        self.pg = placement_group([
            {"CPU": 2, "GPU": 1},  # Worker 1
            {"CPU": 2, "GPU": 1},  # Worker 2
            {"CPU": 4},            # CPU-only worker
        ])
        
        # Wait for resources to be available
        ray.get(self.pg.ready())
        
        # Initialize workers with resource constraints
        self.gpu_workers = [
            self.create_gpu_worker(i) 
            for i in range(2)
        ]
        self.cpu_worker = self.create_cpu_worker()
    
    @ray.remote(num_cpus=2, num_gpus=1)
    def create_gpu_worker(self, worker_id: int):
        class GPUWorker:
            def __init__(self):
                self.model = bentoml.pytorch.load_model(
                    "gpu_model:latest", 
                    device_id=f"cuda:0"
                )
                self.worker_id = worker_id
            
            def predict(self, data):
                return self.model.predict(data)
        
        return GPUWorker()
    
    @ray.remote(num_cpus=4)
    def create_cpu_worker(self):
        class CPUWorker:
            def __init__(self):
                self.model = bentoml.sklearn.load_model("cpu_model:latest")
            
            def predict(self, data):
                return self.model.predict(data)
        
        return CPUWorker()
    
    @bentoml.api
    async def optimized_inference(
        self, 
        gpu_data: List = None, 
        cpu_data: List = None
    ) -> dict:
        futures = []
        
        # Schedule GPU work
        if gpu_data:
            for i, batch in enumerate(gpu_data):
                worker = self.gpu_workers[i % len(self.gpu_workers)]
                future = worker.predict.remote(batch)
                futures.append(("gpu", future))
        
        # Schedule CPU work
        if cpu_data:
            future = self.cpu_worker.predict.remote(cpu_data)
            futures.append(("cpu", future))
        
        # Collect results
        results = {"gpu_results": [], "cpu_results": []}
        for work_type, future in futures:
            result = ray.get(future)
            results[f"{work_type}_results"].append(result)
        
        return results
```

### Fault Tolerance

```python
import ray
import bentoml
from ray.util.actor_pool import ActorPool

@ray.remote(max_restarts=3)
class FaultTolerantWorker:
    def __init__(self, model_tag: str):
        self.model_tag = model_tag
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = bentoml.sklearn.load_model(self.model_tag)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    def predict(self, data):
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Model failed to load")
        
        try:
            return self.model.predict(data)
        except Exception as e:
            # Attempt to reload model on prediction failure
            print(f"Prediction failed, reloading model: {e}")
            self.load_model()
            if self.model:
                return self.model.predict(data)
            raise

@bentoml.service
class FaultTolerantService:
    def __init__(self):
        # Create fault-tolerant worker pool
        self.workers = [
            FaultTolerantWorker.remote("robust_model:latest")
            for _ in range(4)
        ]
        
        self.actor_pool = ActorPool(self.workers)
    
    @bentoml.api
    async def robust_predict(self, input_batches: List[List]) -> List[dict]:
        results = []
        
        # Submit tasks to the pool
        for batch in input_batches:
            future = self.actor_pool.submit(
                lambda actor, batch: actor.predict.remote(batch),
                batch
            )
            results.append(future)
        
        # Collect results with error handling
        final_results = []
        for i, future in enumerate(results):
            try:
                result = ray.get(future)
                final_results.append({
                    "batch_id": i,
                    "success": True,
                    "prediction": result.tolist()
                })
            except Exception as e:
                final_results.append({
                    "batch_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        return final_results
```

## Monitoring and Observability

### Ray Dashboard Integration

```python
import ray
import bentoml
from ray.util.metrics import Counter, Histogram
import time

# Define custom metrics
REQUEST_COUNTER = Counter(
    "bentoml_ray_requests_total",
    description="Total requests processed",
    tag_keys=("method", "status")
)

LATENCY_HISTOGRAM = Histogram(
    "bentoml_ray_request_duration_seconds",
    description="Request processing time",
    boundaries=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

@bentoml.service
class MonitoredRayService:
    def __init__(self):
        # Initialize Ray with dashboard
        if not ray.is_initialized():
            ray.init(include_dashboard=True)
        
        self.model = bentoml.sklearn.load_model("monitored_model:latest")
    
    @bentoml.api
    async def monitored_predict(self, input_data: List[float]) -> dict:
        start_time = time.time()
        
        try:
            prediction = self.model.predict([input_data])
            
            # Record successful request
            REQUEST_COUNTER.inc(tags={"method": "predict", "status": "success"})
            
            processing_time = time.time() - start_time
            LATENCY_HISTOGRAM.observe(processing_time)
            
            return {
                "success": True,
                "prediction": prediction[0],
                "processing_time": processing_time
            }
            
        except Exception as e:
            # Record failed request
            REQUEST_COUNTER.inc(tags={"method": "predict", "status": "error"})
            
            return {
                "success": False,
                "error": str(e)
            }
```

## Best Practices

### 1. Initialization and Cleanup

```python
import ray
import bentoml
import atexit

@bentoml.service
class RayManagedService:
    def __init__(self):
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                address="auto",  # Connect to existing cluster if available
                ignore_reinit_error=True
            )
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
        self.model = bentoml.pytorch.load_model("managed_model:latest")
        self.workers = [
            self.create_worker.remote(i)
            for i in range(4)
        ]
    
    @ray.remote
    def create_worker(self, worker_id: int):
        # Worker initialization logic
        return f"Worker {worker_id} initialized"
    
    def cleanup(self):
        """Cleanup Ray resources"""
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception as e:
            print(f"Error during Ray cleanup: {e}")
    
    @bentoml.api
    async def predict(self, input_data: List[float]) -> dict:
        # Use Ray workers for prediction
        futures = [
            worker.predict.remote(input_data)
            for worker in self.workers
        ]
        
        # Return first successful result
        result = ray.get(futures[0])
        
        return {
            "prediction": result,
            "worker_count": len(self.workers)
        }
```

### 2. Error Handling and Retries

```python
import ray
import bentoml
from ray.exceptions import RayTaskError
import time

@bentoml.service
class ResilientRayService:
    def __init__(self):
        self.model = bentoml.sklearn.load_model("resilient_model:latest")
        self.workers = [
            self.create_resilient_worker.remote("worker_model:latest")
            for _ in range(3)
        ]
    
    @ray.remote(max_retries=3, retry_exceptions=True)
    def create_resilient_worker(self, model_tag: str):
        class ResilientWorker:
            def __init__(self):
                self.model = bentoml.sklearn.load_model(model_tag)
            
            def predict_with_retry(self, data, max_retries=3):
                for attempt in range(max_retries):
                    try:
                        return self.model.predict(data)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
        
        return ResilientWorker()
    
    @bentoml.api
    async def resilient_predict(self, input_data: List[float]) -> dict:
        errors = []
        
        # Try each worker until one succeeds
        for i, worker in enumerate(self.workers):
            try:
                future = worker.predict_with_retry.remote([input_data])
                result = ray.get(future, timeout=30)  # 30 second timeout
                
                return {
                    "success": True,
                    "prediction": result[0],
                    "worker_used": i,
                    "attempts": len(errors) + 1
                }
                
            except (RayTaskError, ray.exceptions.GetTimeoutError) as e:
                errors.append(f"Worker {i}: {str(e)}")
                continue
        
        # All workers failed
        return {
            "success": False,
            "error": "All workers failed",
            "details": errors
        }
```

### 3. Configuration Management

```python
import ray
import bentoml
import os
from typing import Dict, Any

@bentoml.service
class ConfigurableRayService:
    def __init__(self):
        # Ray configuration from environment
        self.ray_config = {
            "address": os.getenv("RAY_ADDRESS", "auto"),
            "num_cpus": int(os.getenv("RAY_NUM_CPUS", "4")),
            "num_gpus": int(os.getenv("RAY_NUM_GPUS", "0")),
            "object_store_memory": int(os.getenv("RAY_OBJECT_STORE_MEMORY", "1000000000"))
        }
        
        # Initialize Ray with configuration
        if not ray.is_initialized():
            ray.init(**self.ray_config)
        
        # Service configuration
        self.model_config = {
            "model_tag": os.getenv("MODEL_TAG", "default_model:latest"),
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "num_workers": int(os.getenv("NUM_WORKERS", "4"))
        }
        
        # Load model and initialize workers
        self.model = bentoml.sklearn.load_model(self.model_config["model_tag"])
        self.workers = self.initialize_workers()
    
    def initialize_workers(self):
        """Initialize workers based on configuration"""
        workers = []
        
        for i in range(self.model_config["num_workers"]):
            worker = self.create_configurable_worker.remote(
                self.model_config["model_tag"],
                self.model_config["batch_size"]
            )
            workers.append(worker)
        
        return workers
    
    @ray.remote
    def create_configurable_worker(self, model_tag: str, batch_size: int):
        class ConfigurableWorker:
            def __init__(self):
                self.model = bentoml.sklearn.load_model(model_tag)
                self.batch_size = batch_size
            
            def predict(self, data):
                return self.model.predict(data)
        
        return ConfigurableWorker()
    
    @bentoml.api
    async def predict(self, input_data: List[float]) -> dict:
        # Distribute work among configured workers
        worker = self.workers[0]  # Simple round-robin could be added
        future = worker.predict.remote([input_data])
        result = ray.get(future)
        
        return {
            "prediction": result[0],
            "configuration": {
                "ray_config": self.ray_config,
                "model_config": self.model_config
            }
        }
```

For more advanced Ray integration patterns and BentoRay guide, refer to the [Ray documentation](https://docs.ray.io/) and the official BentoML Ray integration documentation.