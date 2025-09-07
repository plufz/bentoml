# BentoML Async Task Queues

Guide to implementing background processing and asynchronous task management with BentoML.

## Overview

BentoML async task queues enable **"fire-and-forget"** style inference tasks, allowing you to:
- **Submit background processing requests** without waiting for completion
- **Receive unique task identifiers** immediately for tracking
- **Check task status and retrieve results** asynchronously
- **Improve system responsiveness** by handling long-running operations in the background

## Ideal Use Cases

- **Batch processing** large volumes of data
- **Asynchronous media generation** (images, videos, audio)
- **Time-insensitive tasks** with lower priority requirements
- **Resource-intensive operations** that shouldn't block immediate responses
- **Background data processing** and analysis

## Task Endpoint Architecture

Using the `@bentoml.task` decorator automatically generates multiple endpoints for comprehensive task management:

1. **`POST /submit`** - Queue a new task
2. **`GET /status`** - Check task status and progress
3. **`GET /get`** - Retrieve completed task results
4. **`POST /cancel`** - Cancel pending tasks
5. **`PUT /retry`** - Retry failed tasks

## Basic Task Implementation

### Simple Task Service

```python
import bentoml
from PIL import Image
import time

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
class ImageGenerationService:
    def __init__(self):
        # Load your image generation model
        self.model = self.load_model()
    
    @bentoml.task
    def generate_image(self, prompt: str, steps: int = 20) -> dict:
        """Long-running image generation task"""
        
        # Simulate time-intensive processing
        time.sleep(2)  # Replace with actual model inference
        
        # Generate image using your model
        image = self.model.generate(prompt, num_steps=steps)
        
        return {
            "prompt": prompt,
            "steps": steps,
            "image_path": "/path/to/generated/image.png",
            "generation_time": 2.0
        }
    
    @bentoml.api
    def quick_info(self) -> dict:
        """Regular API endpoint for immediate responses"""
        return {
            "service": "Image Generation",
            "status": "ready",
            "queue_available": True
        }
```

### Advanced Task with Progress Tracking

```python
import bentoml
from typing import Iterator
import logging

@bentoml.service
class BatchProcessingService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @bentoml.task
    def process_batch(self, data_batch: list[dict]) -> dict:
        """Process a batch of data with progress tracking"""
        
        results = []
        total_items = len(data_batch)
        
        self.logger.info(f"Starting batch processing of {total_items} items")
        
        for i, item in enumerate(data_batch):
            # Process individual item
            processed_item = self.process_single_item(item)
            results.append(processed_item)
            
            # Log progress
            progress = (i + 1) / total_items * 100
            self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{total_items})")
            
            # Optional: Add progress to task metadata
            # This would be visible in status checks
            
        self.logger.info("Batch processing completed")
        
        return {
            "total_processed": total_items,
            "results": results,
            "success_count": len([r for r in results if r.get("success", True)]),
            "processing_summary": "Batch completed successfully"
        }
    
    def process_single_item(self, item: dict) -> dict:
        """Process a single data item"""
        try:
            # Your processing logic here
            processed_data = {"processed": True, "data": item}
            return {"success": True, "result": processed_data}
        except Exception as e:
            return {"success": False, "error": str(e), "item": item}
```

## Client Usage

### Python Client Integration

```python
import bentoml
import time

# Create client
client = bentoml.SyncHTTPClient("http://localhost:3000")

# Submit a task
task_id = client.generate_image.submit(
    prompt="A beautiful sunset over mountains",
    steps=50
)

print(f"Task submitted with ID: {task_id}")

# Check task status
while True:
    status = client.generate_image.get_status(task_id)
    print(f"Task status: {status}")
    
    if status in ["SUCCESS", "FAILURE"]:
        break
    
    time.sleep(1)  # Wait before checking again

# Retrieve results
if status == "SUCCESS":
    result = client.generate_image.get(task_id)
    print(f"Task completed: {result}")
else:
    print("Task failed")
```

### Async Client Usage

```python
import asyncio
import bentoml

async def async_task_example():
    client = bentoml.AsyncHTTPClient("http://localhost:3000")
    
    # Submit multiple tasks concurrently
    tasks = []
    for i in range(5):
        task_id = await client.generate_image.submit(
            prompt=f"Image {i+1}: Abstract art",
            steps=20
        )
        tasks.append(task_id)
    
    print(f"Submitted {len(tasks)} tasks")
    
    # Wait for all tasks to complete
    results = []
    for task_id in tasks:
        while True:
            status = await client.generate_image.get_status(task_id)
            if status == "SUCCESS":
                result = await client.generate_image.get(task_id)
                results.append(result)
                break
            elif status == "FAILURE":
                print(f"Task {task_id} failed")
                break
            
            await asyncio.sleep(0.5)
    
    print(f"Completed {len(results)} tasks")
    return results

# Run async example
asyncio.run(async_task_example())
```

## Task Management Features

### Task Status Monitoring

```python
@bentoml.service
class MonitoredTaskService:
    @bentoml.task
    def monitored_task(self, data: str) -> dict:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("Task started")
        
        try:
            # Simulate processing with status updates
            for step in ["preprocessing", "processing", "postprocessing"]:
                logger.info(f"Current step: {step}")
                time.sleep(1)  # Simulate work
            
            result = {"status": "completed", "data_length": len(data)}
            logger.info("Task completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Task failed: {e}")
            raise
```

### Task Cancellation

```python
# Client-side task cancellation
client = bentoml.SyncHTTPClient("http://localhost:3000")

# Submit task
task_id = client.long_running_task.submit(data="large dataset")

# Cancel task if needed
try:
    client.long_running_task.cancel(task_id)
    print(f"Task {task_id} cancelled")
except Exception as e:
    print(f"Could not cancel task: {e}")
```

### Task Retry

```python
# Retry failed tasks
client = bentoml.SyncHTTPClient("http://localhost:3000")

# Check if task failed
status = client.my_task.get_status(task_id)
if status == "FAILURE":
    # Retry the task
    try:
        client.my_task.retry(task_id)
        print(f"Task {task_id} queued for retry")
    except Exception as e:
        print(f"Could not retry task: {e}")
```

## Configuration Options

### Service-Level Configuration

```python
@bentoml.service(
    resources={"cpu": "4", "memory": "8Gi"},
    traffic={"concurrency": 10}  # Limit concurrent tasks
)
class ConfiguredTaskService:
    @bentoml.task(
        retry=3,              # Retry failed tasks up to 3 times
        timeout=300           # Task timeout in seconds
    )
    def configured_task(self, data: str) -> dict:
        # Task implementation
        return {"processed": data}
```

### Queue Configuration

```python
# Configure task queue behavior
@bentoml.service(
    workers=4,  # Number of worker processes for tasks
)
class QueueConfiguredService:
    @bentoml.task
    def queue_task(self, item: dict) -> dict:
        # Background processing
        return {"result": item}
```

## Advanced Patterns

### Task Chaining

```python
@bentoml.service
class ChainedTaskService:
    @bentoml.task
    def step_one(self, data: str) -> dict:
        processed = {"step1": f"processed_{data}"}
        
        # Trigger next step
        self.step_two.submit(processed)
        
        return processed
    
    @bentoml.task  
    def step_two(self, data: dict) -> dict:
        final_result = {"step2": f"final_{data['step1']}"}
        return final_result
```

### Batch Task Processing

```python
@bentoml.service
class BatchTaskService:
    @bentoml.task
    def process_file_batch(self, file_paths: list[str]) -> dict:
        """Process multiple files in background"""
        
        results = []
        for file_path in file_paths:
            try:
                # Process individual file
                result = self.process_single_file(file_path)
                results.append({"file": file_path, "result": result, "success": True})
            except Exception as e:
                results.append({"file": file_path, "error": str(e), "success": False})
        
        successful = len([r for r in results if r["success"]])
        
        return {
            "total_files": len(file_paths),
            "successful": successful,
            "failed": len(file_paths) - successful,
            "results": results
        }
    
    def process_single_file(self, file_path: str) -> dict:
        # File processing logic
        return {"processed_file": file_path}
```

### Priority Task Queues

```python
@bentoml.service
class PriorityTaskService:
    @bentoml.task
    def high_priority_task(self, urgent_data: str) -> dict:
        """High priority task processed first"""
        return {"priority": "high", "data": urgent_data}
    
    @bentoml.task
    def low_priority_task(self, batch_data: str) -> dict:
        """Low priority task for batch processing"""
        return {"priority": "low", "data": batch_data}
```

## Error Handling and Monitoring

### Comprehensive Error Handling

```python
@bentoml.service
class RobustTaskService:
    @bentoml.task
    def robust_task(self, data: dict) -> dict:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input
            if not data or "required_field" not in data:
                raise ValueError("Invalid input data")
            
            # Process data
            logger.info(f"Processing task with data: {data}")
            result = self.process_data(data)
            
            # Validate output
            if not result:
                raise RuntimeError("Processing failed to generate result")
            
            logger.info("Task completed successfully")
            return {"success": True, "result": result}
            
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            return {"success": False, "error": f"Invalid input: {str(e)}"}
        
        except RuntimeError as e:
            logger.error(f"Processing error: {e}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def process_data(self, data: dict) -> dict:
        # Your data processing logic
        return {"processed": True, "input": data}
```

## Best Practices

1. **Task Design**:
   - Keep tasks focused on single responsibilities
   - Include comprehensive error handling
   - Log progress for long-running tasks
   - Return structured results with success indicators

2. **Resource Management**:
   - Configure appropriate resources for task workers
   - Monitor queue length and processing times
   - Implement task timeouts for resource protection

3. **Client Integration**:
   - Handle task failures gracefully
   - Implement exponential backoff for status polling
   - Store task IDs for later retrieval

4. **Monitoring and Debugging**:
   - Use structured logging for task tracking
   - Implement health checks for task queues
   - Monitor task success/failure rates

5. **Production Considerations**:
   - Configure task result retention periods
   - Implement task cleanup mechanisms
   - Set up alerting for failed tasks
   - Plan for task queue scaling

Async task queues in BentoML provide a powerful foundation for building responsive, scalable AI applications that can handle background processing efficiently while maintaining excellent user experience.