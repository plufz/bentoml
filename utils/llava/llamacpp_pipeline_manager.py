"""
LLaVA pipeline manager using llama-cpp-python for GGUF models.
Provides faster inference for quantized LLaVA models.
"""

import torch
import pathlib
import json
from typing import Dict, Any, Optional, Union
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
from llama_cpp.llama_types import ChatCompletionRequestResponseFormat

from .image_processing import process_image_input
from .json_schema import validate_json_schema


class LLaVALlamaCppPipelineManager:
    """
    Pipeline manager for LLaVA GGUF models using llama-cpp-python.
    
    Provides faster inference compared to transformers for quantized models.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        n_ctx: int = 4096,
    ):
        """
        Initialize LLaVA llama-cpp pipeline manager.
        
        Args:
            model_path: Path to GGUF model file
            clip_model_path: Path to CLIP model GGUF file
            device: Device to use (auto-detected if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            n_ctx: Context size
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Auto-detect model paths if not provided
        if model_path is None or clip_model_path is None:
            model_path, clip_model_path = self._get_default_model_paths()
        
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        
        # Device detection and configuration
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
            
        # Configure GPU layers based on device
        if self.device == "cuda":
            n_gpu_layers = -1  # Offload all layers to GPU
        elif self.device == "mps":
            n_gpu_layers = -1  # Offload all layers to MPS
        else:
            n_gpu_layers = 0   # Use CPU only
        
        print(f"Loading LLaVA GGUF model on {self.device}")
        
        # Initialize chat handler and model
        chat_handler = Llava16ChatHandler(clip_model_path=clip_model_path, verbose=False)
        self.llama = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            verbose=False,
            n_gpu_layers=n_gpu_layers,
            use_mlock=True,
        )
        
        print(f"LLaVA GGUF model loaded successfully on {self.device}")
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_default_model_paths(self) -> tuple[str, str]:
        """Get default model paths from HuggingFace cache using HF_HOME environment variable."""
        import os
        import glob
        
        # Use HF_HOME environment variable, fallback to default if not set
        HUGGINGFACE_PATH = os.getenv('HF_HOME', str(pathlib.Path.home()) + "/.cache/huggingface")
        
        # Look for the model directory pattern
        model_dir_pattern = f"{HUGGINGFACE_PATH}/hub/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/*"
        model_dirs = glob.glob(model_dir_pattern)
        
        if not model_dirs:
            raise FileNotFoundError(
                f"LLaVA GGUF model not found in {HUGGINGFACE_PATH}. "
                "Please download the model first or provide explicit model paths."
            )
        
        # Use the first (most recent) snapshot
        snapshot_dir = model_dirs[0]
        
        clip_model_path = f"{snapshot_dir}/mmproj-model-f16.gguf"
        model_path = f"{snapshot_dir}/llava-v1.6-mistral-7b.Q5_K_M.gguf"
        
        # Verify both files exist
        if not os.path.exists(clip_model_path):
            raise FileNotFoundError(f"CLIP model not found: {clip_model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLaVA model not found: {model_path}")
        
        return model_path, clip_model_path
    
    def generate_structured_response(
        self,
        image: Union[str, bytes, Image.Image],
        prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        include_raw_response: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response from image and prompt.
        
        Args:
            image: Image input (URL, base64, bytes, or PIL Image)
            prompt: Text prompt/question about the image
            json_schema: Optional JSON schema for structured output
            include_raw_response: Whether to include raw model response
            
        Returns:
            Dict containing parsed JSON response and metadata
        """
        try:
            # Process image input and convert to data URL
            processed_image = process_image_input(image)
            image_url = self._image_to_data_url(processed_image)
            
            # Determine if we need structured JSON output
            if json_schema:
                return self._generate_json_response(
                    image_url, prompt, json_schema, include_raw_response
                )
            else:
                return self._generate_text_response(
                    image_url, prompt, include_raw_response
                )
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "format": "error",
                "device": self.device,
            }
    
    def _generate_json_response(
        self,
        image_url: str,
        prompt: str,
        json_schema: Dict[str, Any],
        include_raw_response: bool
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        # Validate schema
        if not validate_json_schema(json_schema):
            return {
                "success": False,
                "error": "Invalid JSON schema provided",
                "format": "error",
            }
        
        # Prepare response format for llama-cpp
        response_format = {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            }
        }
        
        system_prompt = "You are a helpful assistant that outputs in JSON format. Analyze the image and respond with JSON data matching the required schema."
        
        # Generate response
        result = self.llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            response_format=response_format,
        )
        
        response_text = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            parsed_json = json.loads(response_text)
            
            response_data = {
                "success": True,
                "prompt": prompt,
                "response": parsed_json,
                "format": "structured_json",
                "device": self.device,
            }
            
            if include_raw_response:
                response_data["raw_response"] = response_text
                
            return response_data
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "raw_response": response_text,
                "format": "error",
                "device": self.device,
            }
    
    def _generate_text_response(
        self,
        image_url: str,
        prompt: str,
        include_raw_response: bool
    ) -> Dict[str, Any]:
        """Generate raw text response."""
        result = self.llama.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        
        response_text = result["choices"][0]["message"]["content"]
        
        response_data = {
            "success": True,
            "prompt": prompt,
            "response": response_text,
            "format": "raw_text",
            "device": self.device,
        }
        
        if include_raw_response:
            response_data["raw_response"] = response_text
            
        return response_data
    
    def _image_to_data_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        import base64
        import io
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Resize for efficiency (similar to the example service)
        image = image.resize((336, 336))
        
        # Convert to base64 data URL
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_data = buffer.getvalue()
        
        return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status information."""
        return {
            "status": "healthy",
            "device": self.device,
            "model": "llava-1.6-mistral-7b-gguf",
            "dtype": "GGUF quantized",
            "mode": "llama-cpp",
        }