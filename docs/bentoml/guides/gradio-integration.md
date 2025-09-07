# Adding Gradio UI to BentoML Services

Guide to integrating Gradio web interfaces with BentoML services for interactive model demos and user interfaces.

## Overview

Gradio integration allows you to:
- **Quickly build web UIs** for AI models
- **Provide interactive demos** alongside API endpoints
- **Test models visually** during development
- **Create user-friendly interfaces** for non-technical users

## Prerequisites

Install the required dependencies:

```bash
pip install bentoml gradio fastapi
```

## Basic Integration

### Simple Text Processing Service

```python
import bentoml
import gradio as gr
from transformers import pipeline

# Helper function to call service method
def process_text(text: str) -> str:
    svc_instance = bentoml.get_current_service()
    result = svc_instance.summarize(text)
    return result

# Create Gradio interface
interface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter text to summarize...",
            lines=10
        )
    ],
    outputs=[
        gr.Textbox(
            label="Summary",
            lines=5
        )
    ],
    title="Text Summarization Service",
    description="Enter text to get an AI-generated summary"
)

@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"}
)
@bentoml.gradio.mount_gradio_app(interface, path="/ui")
class TextSummarization:
    def __init__(self):
        self.model = pipeline("summarization", model="facebook/bart-base")
    
    @bentoml.api
    def summarize(self, text: str) -> str:
        """API endpoint for text summarization"""
        result = self.model(text, max_length=150, min_length=30)
        return result[0]["summary_text"]
```

## Advanced UI Components

### Image Processing Service

```python
import bentoml
import gradio as gr
from PIL import Image
import numpy as np

def process_image(image: Image.Image, filter_type: str) -> Image.Image:
    svc_instance = bentoml.get_current_service()
    return svc_instance.apply_filter(image, filter_type)

# Advanced Gradio interface with multiple input types
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(
            type="pil",
            label="Upload Image"
        ),
        gr.Dropdown(
            choices=["blur", "sharpen", "edge_detect", "vintage"],
            value="blur",
            label="Filter Type"
        )
    ],
    outputs=[
        gr.Image(
            type="pil",
            label="Processed Image"
        )
    ],
    title="Image Filter Service",
    description="Apply various filters to your images",
    examples=[
        ["example1.jpg", "blur"],
        ["example2.jpg", "vintage"]
    ]
)

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
@bentoml.gradio.mount_gradio_app(interface, path="/demo")
class ImageProcessor:
    def __init__(self):
        # Load image processing models
        self.filters = self.load_filters()
    
    @bentoml.api
    def apply_filter(self, image: Image.Image, filter_type: str) -> Image.Image:
        """Apply specified filter to image"""
        # Filter implementation
        processed = self.filters[filter_type](image)
        return processed
```

### Multi-Modal Service

```python
import bentoml
import gradio as gr
from typing import Dict, Any

def analyze_content(image: Image.Image, text: str, analysis_type: str) -> Dict[str, Any]:
    svc_instance = bentoml.get_current_service()
    return svc_instance.analyze(image, text, analysis_type)

# Multi-input Gradio interface
interface = gr.Interface(
    fn=analyze_content,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Description", lines=3),
        gr.Radio(
            choices=["similarity", "sentiment", "content_match"],
            value="similarity",
            label="Analysis Type"
        )
    ],
    outputs=[
        gr.JSON(label="Analysis Results"),
        gr.Textbox(label="Summary")
    ],
    title="Multi-Modal Analyzer",
    description="Analyze image-text relationships"
)

@bentoml.service
@bentoml.gradio.mount_gradio_app(interface, path="/analyze")
class MultiModalAnalyzer:
    @bentoml.api
    def analyze(self, image: Image.Image, text: str, analysis_type: str) -> Dict[str, Any]:
        # Multi-modal analysis implementation
        return {
            "similarity_score": 0.85,
            "sentiment": "positive",
            "analysis_type": analysis_type
        }
```

## Custom Interface Patterns

### Tabbed Interface

```python
import gradio as gr

# Create tabbed interface for multiple features
with gr.Blocks() as interface:
    gr.Markdown("# AI Service Demo")
    
    with gr.Tab("Text Processing"):
        text_input = gr.Textbox(label="Input Text")
        text_output = gr.Textbox(label="Processed Text")
        text_btn = gr.Button("Process Text")
        
        text_btn.click(
            fn=process_text,
            inputs=[text_input],
            outputs=[text_output]
        )
    
    with gr.Tab("Image Processing"):
        image_input = gr.Image(type="pil", label="Input Image")
        image_output = gr.Image(type="pil", label="Processed Image")
        image_btn = gr.Button("Process Image")
        
        image_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[image_output]
        )

@bentoml.service
@bentoml.gradio.mount_gradio_app(interface, path="/demo")
class MultiFeatureService:
    # Service implementation
    pass
```

### Interactive Chat Interface

```python
import gradio as gr

def chat_response(message: str, history: list) -> tuple:
    svc_instance = bentoml.get_current_service()
    response = svc_instance.chat(message, history)
    
    # Update chat history
    history.append((message, response))
    return history, ""

# Chat interface
with gr.Blocks() as chat_interface:
    gr.Markdown("# AI Chat Assistant")
    
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(
        label="Message",
        placeholder="Type your message here..."
    )
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(
        fn=chat_response,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

@bentoml.service
@bentoml.gradio.mount_gradio_app(chat_interface, path="/chat")
class ChatService:
    @bentoml.api
    def chat(self, message: str, history: list = None) -> str:
        # Chat implementation
        return f"Response to: {message}"
```

## Configuration Options

### Custom Styling

```python
import gradio as gr

# Custom CSS styling
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.title {
    color: #2196F3;
    text-align: center;
}
"""

interface = gr.Interface(
    fn=process_function,
    inputs=[...],
    outputs=[...],
    title="Styled AI Service",
    css=css,
    theme=gr.themes.Soft()
)
```

### Authentication

```python
interface = gr.Interface(
    fn=process_function,
    inputs=[...],
    outputs=[...],
    title="Protected Service",
    auth=("username", "password")  # Simple auth
)

# Or with custom auth function
def auth_function(username: str, password: str) -> bool:
    return username == "admin" and password == "secret"

interface = gr.Interface(
    fn=process_function,
    inputs=[...],
    outputs=[...],
    auth=auth_function
)
```

### API Integration

```python
# Enable API alongside Gradio UI
interface = gr.Interface(
    fn=process_function,
    inputs=[...],
    outputs=[...],
    api_name="process",  # Custom API endpoint name
    allow_flagging=False
)

@bentoml.service
@bentoml.gradio.mount_gradio_app(interface, path="/ui", api_name="gradio_api")
class ServiceWithGradio:
    @bentoml.api
    def api_endpoint(self, data: str) -> str:
        """Regular API endpoint"""
        return self.process(data)
    
    def process(self, data: str) -> str:
        # Shared processing logic
        return f"Processed: {data}"
```

## Deployment and Access

### Running the Service

```bash
# Start the service
bentoml serve service.py:ServiceName

# Access points:
# - API: http://localhost:3000/
# - Gradio UI: http://localhost:3000/ui
# - Health check: http://localhost:3000/health
```

### Production Considerations

```python
@bentoml.service(
    resources={"cpu": "4", "memory": "8Gi"},
    traffic={"concurrency": 10}
)
@bentoml.gradio.mount_gradio_app(
    interface, 
    path="/ui",
    auth=auth_function,  # Add authentication
    show_api=False       # Hide API docs in production
)
class ProductionService:
    # Service implementation
    pass
```

## Error Handling

```python
def safe_process(text: str) -> str:
    try:
        svc_instance = bentoml.get_current_service()
        result = svc_instance.process(text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=safe_process,
    inputs=[gr.Textbox(label="Input")],
    outputs=[gr.Textbox(label="Output")],
    title="Safe Processing Service"
)
```

## Best Practices

1. **Keep UI functions simple** - Handle complex logic in the service methods
2. **Use appropriate input/output components** - Match Gradio components to your data types
3. **Add helpful descriptions** - Include titles, labels, and placeholders
4. **Handle errors gracefully** - Wrap functions in try-catch blocks
5. **Consider authentication** - Add auth for production deployments
6. **Test thoroughly** - Test both UI and API endpoints
7. **Use examples** - Provide example inputs for better user experience

## Complete Example

```python
import bentoml
import gradio as gr
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_image(image: Image.Image) -> dict:
    """UI function for image classification"""
    try:
        svc_instance = bentoml.get_current_service()
        result = svc_instance.classify(image)
        logger.info(f"Classification completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"error": str(e)}

# Create comprehensive UI
interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(
            type="pil",
            label="Upload Image to Classify"
        )
    ],
    outputs=[
        gr.JSON(label="Classification Results")
    ],
    title="Image Classification Service",
    description="Upload an image to get AI-powered classification results",
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"],
        ["examples/car.jpg"]
    ],
    cache_examples=True,
    allow_flagging="never"
)

@bentoml.service(
    resources={"gpu": "1", "memory": "4Gi"}
)
@bentoml.gradio.mount_gradio_app(interface, path="/classify")
class ImageClassificationService:
    def __init__(self):
        # Load classification model
        self.model = self.load_model()
        logger.info("Image classification service initialized")
    
    def load_model(self):
        # Load your image classification model
        logger.info("Loading classification model...")
        return None  # Replace with actual model loading
    
    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        """Classify uploaded image"""
        try:
            # Perform classification
            # result = self.model.predict(image)
            
            # Mock result for example
            result = {
                "predictions": [
                    {"label": "cat", "confidence": 0.95},
                    {"label": "dog", "confidence": 0.03},
                    {"label": "bird", "confidence": 0.02}
                ],
                "top_prediction": "cat",
                "confidence": 0.95
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise
```

This comprehensive example demonstrates a production-ready service with both API and Gradio UI endpoints, proper error handling, and logging.