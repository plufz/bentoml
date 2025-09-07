"""
Pipeline manager for Real-ESRGAN upscaling models
"""

import torch
from typing import Optional
from PIL import Image
import numpy as np


class UpscalerPipelineManager:
    """Manages Real-ESRGAN model loading and inference"""
    
    def __init__(self):
        self.device = self._detect_device()
        self.model = None
        self.face_enhancer = None
        self.model_name = "Real-ESRGAN x4plus"
        
    def _detect_device(self) -> str:
        """Detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  
        else:
            return "cpu"
    
    def _load_model(self):
        """Lazy load the Real-ESRGAN model"""
        if self.model is None:
            try:
                from realesrgan import RealESRGAN
                
                self.model = RealESRGAN(self.device, scale=4)
                self.model.load_weights('weights/RealESRGAN_x4plus.pth', download=True)
                
                print(f"Real-ESRGAN model loaded on {self.device}")
                
            except ImportError:
                print("Warning: realesrgan not available, using PIL fallback")
                self.model = "fallback"
                self.model_name = "PIL Lanczos (fallback)"
            except Exception as e:
                print(f"Warning: Failed to load Real-ESRGAN: {e}")
                self.model = "fallback"
                self.model_name = "PIL Lanczos (fallback)"
    
    def _load_face_enhancer(self):
        """Lazy load GFPGAN face enhancement model"""
        if self.face_enhancer is None:
            try:
                from gfpgan import GFPGANer
                
                self.face_enhancer = GFPGANer(
                    model_path='weights/GFPGANv1.4.pth',
                    upscale=1,
                    arch='clean', 
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                print("GFPGAN face enhancer loaded")
                
            except ImportError:
                print("Warning: GFPGAN not available for face enhancement")
                self.face_enhancer = "unavailable"
            except Exception as e:
                print(f"Warning: Failed to load face enhancer: {e}")
                self.face_enhancer = "unavailable"
    
    def upscale_image(
        self, 
        image: Image.Image, 
        scale_factor: float = 2.0,
        face_enhance: bool = False
    ) -> Image.Image:
        """
        Upscale an image using Real-ESRGAN or fallback method
        
        Args:
            image: PIL Image to upscale
            scale_factor: Scaling factor (1.0-4.0)
            face_enhance: Whether to apply face enhancement
            
        Returns:
            Upscaled PIL Image
        """
        self._load_model()
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Fallback upscaling using PIL
        if self.model == "fallback":
            new_size = (
                int(image.size[0] * scale_factor),
                int(image.size[1] * scale_factor)
            )
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        try:
            # Convert PIL to numpy array (RGB)
            img_array = np.array(image)
            
            # Real-ESRGAN expects BGR format
            img_bgr = img_array[:, :, ::-1]  # RGB to BGR
            
            # Upscale with Real-ESRGAN (always 4x)
            upscaled_bgr = self.model.predict(img_bgr)
            
            # Convert back to RGB
            upscaled_rgb = upscaled_bgr[:, :, ::-1]  # BGR to RGB
            upscaled_image = Image.fromarray(upscaled_rgb)
            
            # Apply face enhancement if requested
            if face_enhance:
                upscaled_image = self._enhance_faces(upscaled_image)
            
            # Adjust to requested scale factor if different from 4x
            if abs(scale_factor - 4.0) > 0.1:
                target_size = (
                    int(image.size[0] * scale_factor),
                    int(image.size[1] * scale_factor)
                )
                upscaled_image = upscaled_image.resize(target_size, Image.Resampling.LANCZOS)
            
            return upscaled_image
            
        except Exception as e:
            print(f"Real-ESRGAN failed: {e}, falling back to PIL")
            # Fallback to PIL upscaling
            new_size = (
                int(image.size[0] * scale_factor),
                int(image.size[1] * scale_factor)
            )
            return image.resize(new_size, Image.Resampling.LANCZOS)
    
    def _enhance_faces(self, image: Image.Image) -> Image.Image:
        """Apply face enhancement using GFPGAN"""
        self._load_face_enhancer()
        
        if self.face_enhancer == "unavailable":
            return image
        
        try:
            # Convert to numpy array (BGR for GFPGAN)
            img_array = np.array(image)
            img_bgr = img_array[:, :, ::-1]  # RGB to BGR
            
            # Apply face enhancement
            _, _, enhanced_bgr = self.face_enhancer.enhance(
                img_bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
            
            # Convert back to RGB
            enhanced_rgb = enhanced_bgr[:, :, ::-1]  # BGR to RGB
            return Image.fromarray(enhanced_rgb)
            
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return image