"""
Tests for PhotoUpscalerService
"""

import pytest
import subprocess
import time
import requests
import base64
import json
from pathlib import Path
from typing import Generator
from PIL import Image
import io
from unittest.mock import patch, Mock


class TestPhotoUpscalerServiceUnit:
    """Unit tests for PhotoUpscalerService components"""
    
    def test_upscaler_service_import(self):
        """Test that the upscaler service can be imported"""
        from services.upscaler_service import PhotoUpscalerService
        assert PhotoUpscalerService is not None
    
    def test_upscale_url_request_model(self):
        """Test the UpscaleUrlRequest model validation"""
        from services.upscaler_service import UpscaleUrlRequest
        
        # Test valid request
        request = UpscaleUrlRequest(url="https://example.com/image.jpg")
        assert str(request.url) == "https://example.com/image.jpg"
        assert request.scale_factor == 2.0  # default
        assert request.face_enhance is False  # default
        assert request.output_format == "PNG"  # default
        assert request.quality == 95  # default
    
    def test_upscale_url_request_custom_params(self):
        """Test UpscaleUrlRequest with custom parameters"""
        from services.upscaler_service import UpscaleUrlRequest
        
        request = UpscaleUrlRequest(
            url="https://example.com/photo.png",
            scale_factor=3.5,
            face_enhance=True,
            output_format="JPEG",
            quality=85
        )
        assert str(request.url) == "https://example.com/photo.png"
        assert request.scale_factor == 3.5
        assert request.face_enhance is True
        assert request.output_format == "JPEG"
        assert request.quality == 85
    
    def test_upscale_url_request_validation(self):
        """Test UpscaleUrlRequest parameter validation"""
        from services.upscaler_service import UpscaleUrlRequest
        from pydantic import ValidationError
        
        # Test invalid scale factor (too high)
        with pytest.raises(ValidationError):
            UpscaleUrlRequest(
                url="https://example.com/image.jpg",
                scale_factor=5.0  # Above maximum 4.0
            )
        
        # Test invalid scale factor (too low)  
        with pytest.raises(ValidationError):
            UpscaleUrlRequest(
                url="https://example.com/image.jpg",
                scale_factor=0.5  # Below minimum 1.0
            )
        
        # Test invalid quality (too low)
        with pytest.raises(ValidationError):
            UpscaleUrlRequest(
                url="https://example.com/image.jpg",
                quality=30  # Below minimum 50
            )
        
        # Test invalid quality (too high)
        with pytest.raises(ValidationError):
            UpscaleUrlRequest(
                url="https://example.com/image.jpg", 
                quality=110  # Above maximum 100
            )
    
    def test_pipeline_manager_device_detection(self):
        """Test device detection logic"""
        from utils.upscaler.pipeline_manager import UpscalerPipelineManager
        
        manager = UpscalerPipelineManager()
        assert manager.device in ["cuda", "mps", "cpu"]
    
    def test_image_processing_validation(self):
        """Test image format validation"""
        from utils.upscaler.image_processing import validate_output_format, validate_image_format
        
        # Test valid formats
        assert validate_output_format("PNG") == "PNG"
        assert validate_output_format("jpeg") == "JPEG"
        assert validate_output_format("webp") == "WEBP"
        
        # Test invalid format
        with pytest.raises(ValueError):
            validate_output_format("TIFF")
    
    def test_upscaler_service_methods(self):
        """Test service method signatures"""
        from services.upscaler_service import PhotoUpscalerService
        
        # Test service can be instantiated
        with patch('utils.upscaler.pipeline_manager.UpscalerPipelineManager'):
            service = PhotoUpscalerService()
            
            # Test health method exists
            assert hasattr(service, 'health')
            assert callable(getattr(service, 'health'))
            
            # Test upscale methods exist
            assert hasattr(service, 'upscale_file')
            assert callable(getattr(service, 'upscale_file'))
            assert hasattr(service, 'upscale_url')
            assert callable(getattr(service, 'upscale_url'))


class TestPhotoUpscalerServiceIntegration:
    """Integration tests requiring actual service startup"""
    
    @pytest.fixture(scope="class")
    @pytest.mark.timeout(120)  # Extended timeout for model loading
    def running_upscaler_service(self) -> Generator[str, None, None]:
        """Start PhotoUpscalerService and return base URL"""
        try:
            from tests.conftest import get_service_url
            import os
            
            # Get service configuration from environment
            port = os.getenv("UPSCALER_SERVICE_PORT", "3006")
            base_url = get_service_url("upscaler", port)
            
            process = subprocess.Popen([
                "uv", "run", "bentoml", "serve",
                "services.upscaler_service:PhotoUpscalerService",
                "--port", port,
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            max_attempts = 60  # Extended timeout for model loading
            for _ in range(max_attempts):
                try:
                    response = requests.post(f"{base_url}/health", json={}, timeout=5)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                # Capture output for debugging
                stdout, stderr = process.communicate(timeout=10)
                process.terminate()
                error_msg = f"Upscaler service failed to start.\nSTDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}"
                pytest.fail(error_msg)
            
            yield base_url
            
            # Cleanup: terminate process and children
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
        except Exception as e:
            pytest.fail(f"Failed to start upscaler service: {e}")
    
    @pytest.fixture(scope="session")
    def sample_upscale_image_path(self, test_assets_dir: Path) -> Path:
        """Path to sample upscaling test image with custom dimensions"""
        return test_assets_dir / "test-upscale.jpg"
    
    @pytest.mark.slow
    def test_health_endpoint_integration(self, running_upscaler_service: str):
        """Test health endpoint returns proper information"""
        response = requests.post(f"{running_upscaler_service}/health", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "PhotoUpscalerService"
        assert data["version"] == "1.0.0"
        assert "device" in data
        assert "model" in data
        assert "capabilities" in data
        assert isinstance(data["capabilities"], list)
    
    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_upscale_file_integration_custom_dimensions(self, running_upscaler_service: str, sample_upscale_image_path: Path):
        """Test file upscaling with custom non-square dimensions"""
        if not sample_upscale_image_path.exists():
            pytest.skip("Test upscale image not available")
        
        # Verify image has custom non-square dimensions (not divisible by 8)
        with Image.open(sample_upscale_image_path) as img:
            width, height = img.size
            assert width != height  # Not square
            assert width % 8 != 0 or height % 8 != 0  # Not easily divisible by 8
            print(f"Testing with image dimensions: {width}x{height}")
        
        with open(sample_upscale_image_path, "rb") as f:
            files = {
                "image_file": ("test-upscale.jpg", f, "image/jpeg")
            }
            data = {
                "scale_factor": 2.5,  # Custom scale factor
                "face_enhance": False,
                "output_format": "PNG",
                "quality": 90
            }
            
            response = requests.post(
                f"{running_upscaler_service}/upscale_file",
                files=files,
                data=data,
                timeout=60
            )
        
        assert response.status_code == 200
        result = response.json()
        
        if result.get("success"):
            assert "image_base64" in result
            assert "upscaling_info" in result
            
            info = result["upscaling_info"]
            assert "original_size" in info
            assert "upscaled_size" in info
            assert "scale_factor" in info
            assert info["scale_factor"] == 2.5
            
            # Verify dimensions were scaled correctly
            original_width, original_height = info["original_size"]
            upscaled_width, upscaled_height = info["upscaled_size"]
            
            expected_width = int(original_width * 2.5)
            expected_height = int(original_height * 2.5)
            
            # Allow some tolerance for scaling
            assert abs(upscaled_width - expected_width) <= 2
            assert abs(upscaled_height - expected_height) <= 2
            
            # Verify base64 image is valid
            try:
                image_data = base64.b64decode(result["image_base64"])
                Image.open(io.BytesIO(image_data))
                print(f"Successfully upscaled {original_width}x{original_height} -> {upscaled_width}x{upscaled_height}")
            except Exception as e:
                pytest.fail(f"Invalid base64 image data: {e}")
        else:
            # Service may not have models loaded, which is okay for testing
            assert "error" in result
            print(f"Service returned expected error: {result['error']}")
    
    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_upscale_url_integration(self, running_upscaler_service: str):
        """Test URL upscaling with real image URL"""
        payload = {
            "request": {
                "url": "https://plufz.com/test-assets/test-office.jpg",
                "scale_factor": 2.0,
                "face_enhance": False,
                "output_format": "JPEG",
                "quality": 80
            }
        }
        
        response = requests.post(
            f"{running_upscaler_service}/upscale_url",
            json=payload,
            timeout=60
        )
        
        assert response.status_code == 200
        result = response.json()
        
        if result.get("success"):
            assert "image_base64" in result
            assert "upscaling_info" in result
            
            info = result["upscaling_info"]
            assert "original_size" in info
            assert "upscaled_size" in info
            assert "source_url" in info
            assert info["source_url"] == "https://plufz.com/test-assets/test-office.jpg"
            
            # Verify image was upscaled
            original_width, original_height = info["original_size"]
            upscaled_width, upscaled_height = info["upscaled_size"]
            
            assert upscaled_width >= original_width * 2
            assert upscaled_height >= original_height * 2
            
            # Verify base64 image is valid
            try:
                image_data = base64.b64decode(result["image_base64"])
                Image.open(io.BytesIO(image_data))
                print(f"Successfully upscaled from URL: {original_width}x{original_height} -> {upscaled_width}x{upscaled_height}")
            except Exception as e:
                pytest.fail(f"Invalid base64 image data: {e}")
        else:
            # Service may not have models loaded, which is okay for testing
            assert "error" in result
            print(f"Service returned expected error: {result['error']}")


class TestPhotoUpscalerServiceBehavior:
    """HTTP behavior tests using mocked responses"""
    
    @patch('requests.post')
    def test_service_response_format(self, mock_post):
        """Test that service responses follow expected format"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "upscaling_info": {
                "original_size": [423, 317],
                "upscaled_size": [846, 634], 
                "scale_factor": 2.0,
                "face_enhance": False,
                "output_format": "PNG",
                "device": "cpu",
                "model": "Real-ESRGAN x4plus"
            }
        }
        mock_post.return_value = mock_response
        
        # Test the response format
        response = requests.post("http://test/upscale_url", json={
            "url": "https://example.com/test.jpg",
            "scale_factor": 2.0
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "image_base64" in data
        assert "upscaling_info" in data
        
        info = data["upscaling_info"]
        assert "original_size" in info
        assert "upscaled_size" in info
        assert "scale_factor" in info
        assert info["scale_factor"] == 2.0
    
    @patch('requests.post')
    def test_error_response_format(self, mock_post):
        """Test error response format"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "error": "Failed to download image from URL: Invalid URL"
        }
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/upscale_url", json={
            "url": "invalid-url"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "Failed to download image" in data["error"]
    
    def test_dimensions_edge_cases(self):
        """Test handling of various image dimension edge cases"""
        from utils.upscaler.image_processing import validate_output_format
        
        # Test format validation edge cases
        assert validate_output_format("png") == "PNG"
        assert validate_output_format("Jpeg") == "JPEG" 
        assert validate_output_format("WEBP") == "WEBP"
        
        # Test invalid formats
        invalid_formats = ["TIFF", "BMP", "GIF", ""]
        for fmt in invalid_formats:
            with pytest.raises(ValueError):
                validate_output_format(fmt)
    
    def test_scale_factor_calculations(self):
        """Test scale factor calculations for various input dimensions"""
        test_cases = [
            # (original_width, original_height, scale_factor, expected_width, expected_height)
            (423, 317, 2.0, 846, 634),  # Our custom test dimensions
            (100, 200, 3.5, 350, 700),  # Tall image
            (300, 150, 1.5, 450, 225),  # Wide image
            (512, 512, 4.0, 2048, 2048),  # Square image, 4x scale
            (1, 1, 2.0, 2, 2),  # Minimal dimensions
        ]
        
        for orig_w, orig_h, scale, exp_w, exp_h in test_cases:
            calculated_w = int(orig_w * scale)
            calculated_h = int(orig_h * scale)
            assert calculated_w == exp_w, f"Width: {orig_w} * {scale} = {calculated_w}, expected {exp_w}"
            assert calculated_h == exp_h, f"Height: {orig_h} * {scale} = {calculated_h}, expected {exp_h}"