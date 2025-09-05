"""
Pipeline manager for LLaVA (Large Language and Vision Assistant) models.
Handles model loading, image+text processing, and structured output generation.
"""

import torch
from typing import Dict, Any, Optional, Union, List
from transformers import pipeline, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

from ..stable_diffusion.device import detect_device, get_optimal_dtype, DeviceType
from ..stable_diffusion.hf_env import setup_hf_environment, print_hf_setup
from .image_processing import process_image_input, prepare_image_for_llava
from .json_schema import create_json_prompt, parse_json_response


class LLaVAPipelineManager:
    """
    Pipeline manager for LLaVA multimodal models.
    
    Handles loading LLaVA models and processing image+text inputs to generate
    structured JSON responses.
    """
    
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        custom_hf_home: Optional[str] = None,
        device: Optional[DeviceType] = None,
        use_pipeline: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """
        Initialize LLaVA pipeline manager.
        
        Args:
            model_id: HuggingFace model identifier
            custom_hf_home: Custom HuggingFace cache directory
            device: Specific device to use (auto-detected if None)
            use_pipeline: Whether to use transformers pipeline (simpler) or manual loading
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (lower = more deterministic)
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_pipeline = use_pipeline
        
        # Setup HuggingFace environment
        hf_home = setup_hf_environment(custom_hf_home)
        print_hf_setup(hf_home)
        
        # Detect and configure device
        self.device = device or detect_device()
        self.dtype = get_optimal_dtype(self.device)
        
        # Load pipeline or model
        if use_pipeline:
            self.pipe = self._load_pipeline()
            self.processor = None
            self.model = None
        else:
            self.pipe = None
            self.processor, self.model = self._load_model_manually()
    
    def _load_pipeline(self):
        """Load LLaVA using transformers pipeline."""
        print(f"Loading {self.model_id} pipeline on {self.device} with {self.dtype}")
        
        device_id = 0 if self.device == "cuda" else -1 if self.device == "cpu" else "mps"
        
        pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            device=device_id if self.device != "mps" else -1,  # Pipeline doesn't support MPS directly
            torch_dtype=self.dtype,
            model_kwargs={
                "low_cpu_mem_usage": True,
            }
        )
        
        # Move to MPS manually if needed
        if self.device == "mps" and hasattr(pipe, 'model'):
            pipe.model = pipe.model.to("mps")
            
        return pipe
    
    def _load_model_manually(self):
        """Load LLaVA model and processor manually for more control."""
        print(f"Loading {self.model_id} manually on {self.device} with {self.dtype}")
        
        processor = LlavaNextProcessor.from_pretrained(self.model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        
        # Move to device
        model = model.to(self.device)
        
        return processor, model
    
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
        # Process image input
        processed_image = process_image_input(image)
        processed_image = prepare_image_for_llava(processed_image)
        
        # Create structured prompt if schema provided
        if json_schema:
            formatted_prompt = create_json_prompt(prompt, json_schema)
        else:
            formatted_prompt = prompt
        
        # Generate response
        if self.use_pipeline:
            raw_response = self._generate_with_pipeline(processed_image, formatted_prompt)
        else:
            raw_response = self._generate_with_model(processed_image, formatted_prompt)
        
        # Parse response
        result = {
            "success": True,
            "prompt": prompt,
            "model": self.model_id,
            "device": self.device,
        }
        
        if include_raw_response:
            result["raw_response"] = raw_response
        
        if json_schema:
            try:
                parsed_json = parse_json_response(raw_response, json_schema)
                result["response"] = parsed_json
                result["format"] = "structured_json"
            except ValueError as e:
                result["success"] = False
                result["error"] = str(e)
                result["raw_response"] = raw_response
                result["format"] = "raw_text"
        else:
            result["response"] = raw_response
            result["format"] = "raw_text"
        
        return result
    
    def _generate_with_pipeline(self, image: Image.Image, prompt: str) -> str:
        """Generate response using transformers pipeline."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        outputs = self.pipe(
            messages, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        
        return outputs[0]["generated_text"][-1]["content"]
    
    def _generate_with_model(self, image: Image.Image, prompt: str) -> str:
        """Generate response using model directly."""
        # Format prompt in LLaVA-1.6-Mistral format
        formatted_prompt = f"[INST] <image>\n{prompt} [/INST]"
        
        # Process inputs
        inputs = self.processor(formatted_prompt, image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode response (exclude input tokens)
        response = self.processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "use_pipeline": self.use_pipeline,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status information."""
        return {
            "status": "healthy",
            "device": self.device,
            "model": self.model_id,
            "dtype": str(self.dtype),
            "mode": "pipeline" if self.use_pipeline else "manual",
        }