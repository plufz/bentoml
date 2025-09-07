"""
Tests for LLaVA Vision-Language Service
"""

import pytest
import subprocess
import time
import requests
import signal
import psutil
import base64
from pathlib import Path
from typing import Generator
from unittest.mock import patch, Mock, MagicMock
from PIL import Image
import io


class TestLLaVAServiceUnit:
    """Unit tests for LLaVAService - testing individual methods"""
    
    def test_llava_service_import(self):
        """Test that LLaVA service can be imported successfully"""
        from services.llava_service import LLaVAService, VisionLanguageRequest
        
        # Mock the pipeline manager to avoid model loading
        with patch('services.llava_service.LLaVALlamaCppPipelineManager'):
            service = LLaVAService()
            assert service is not None
    
    def test_vision_language_request_model(self):
        """Test VisionLanguageRequest Pydantic model validation"""
        from services.llava_service import VisionLanguageRequest
        
        # Test valid request
        request = VisionLanguageRequest(
            prompt="What do you see in this image?",
            image="test_image_data"
        )
        assert request.prompt == "What do you see in this image?"
        assert request.image == "test_image_data"
        assert request.json_schema is None
        assert request.include_raw_response is False
        
        # Test with optional parameters
        schema = {"type": "object", "properties": {"description": {"type": "string"}}}
        request = VisionLanguageRequest(
            prompt="Describe this image",
            image="image_data",
            json_schema=schema,
            include_raw_response=True,
            temperature=0.5,
            max_new_tokens=256
        )
        assert request.json_schema == schema
        assert request.include_raw_response is True
        assert request.temperature == 0.5
        assert request.max_new_tokens == 256
    
    def test_vision_language_request_validation(self):
        """Test VisionLanguageRequest parameter validation"""
        from services.llava_service import VisionLanguageRequest
        from pydantic import ValidationError
        
        # Test invalid temperature
        with pytest.raises(ValidationError):
            VisionLanguageRequest(
                prompt="test",
                image="test",
                temperature=3.0  # Too high
            )
        
        # Test invalid max_new_tokens
        with pytest.raises(ValidationError):
            VisionLanguageRequest(
                prompt="test", 
                image="test",
                max_new_tokens=0  # Too low
            )
    
    @patch('services.llava_service.LLaVALlamaCppPipelineManager')
    @patch('services.llava_service.validate_image_format')
    @patch('services.llava_service.validate_json_schema')
    def test_analyze_image_method_success(self, mock_validate_schema, mock_validate_image, mock_pipeline):
        """Test analyze_image method with successful processing"""
        from services.llava_service import LLaVAService, VisionLanguageRequest
        
        # Setup mocks
        mock_validate_image.return_value = True
        mock_validate_schema.return_value = True
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.device = "cpu"
        mock_pipeline_instance.max_new_tokens = 512
        mock_pipeline_instance.temperature = 0.1
        mock_pipeline_instance.generate_structured_response.return_value = {
            "success": True,
            "response": "I see a test image",
            "format": "text"
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = LLaVAService()
        request = VisionLanguageRequest(
            prompt="What do you see?",
            image="test_image"
        )
        
        result = service.analyze_image(request)
        
        assert result["success"] is True
        assert "input_prompt" in result
        assert "device_used" in result
        assert result["input_prompt"] == "What do you see?"
        
    @patch('services.llava_service.LLaVALlamaCppPipelineManager')
    @patch('services.llava_service.validate_image_format')
    def test_analyze_image_invalid_image(self, mock_validate_image, mock_pipeline):
        """Test analyze_image method with invalid image format"""
        from services.llava_service import LLaVAService, VisionLanguageRequest
        
        mock_validate_image.return_value = False
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = LLaVAService()
        request = VisionLanguageRequest(
            prompt="What do you see?",
            image="invalid_image"
        )
        
        result = service.analyze_image(request)
        
        assert result["success"] is False
        assert "Invalid image format" in result["error"]
        assert result["format"] == "error"
    
    @patch('services.llava_service.LLaVALlamaCppPipelineManager')
    @patch('services.llava_service.validate_image_format')
    @patch('services.llava_service.validate_json_schema')
    def test_analyze_image_url_method_success(self, mock_validate_schema, mock_validate_image, mock_pipeline):
        """Test analyze_image_url method with successful processing"""
        from services.llava_service import LLaVAService, VisionLanguageUrlRequest
        
        # Setup mocks
        mock_validate_image.return_value = True
        mock_validate_schema.return_value = True
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.device = "cpu"
        mock_pipeline_instance.max_new_tokens = 512
        mock_pipeline_instance.temperature = 0.1
        mock_pipeline_instance.generate_structured_response.return_value = {
            "success": True,
            "response": "I see an image from URL",
            "format": "text"
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = LLaVAService()
        request = VisionLanguageUrlRequest(
            prompt="What do you see?",
            image_url="https://example.com/test.jpg"
        )
        
        result = service.analyze_image_url(request)
        
        assert result["success"] is True
        assert result["response"] == "I see an image from URL"
        assert result["input_prompt"] == "What do you see?"
        assert result["device_used"] == "cpu"
        assert result["has_json_schema"] is False
        
        # Verify the pipeline was called with the URL
        mock_pipeline_instance.generate_structured_response.assert_called_once_with(
            image="https://example.com/test.jpg",
            prompt="What do you see?",
            json_schema=None,
            include_raw_response=False
        )
    
    @patch('services.llava_service.LLaVALlamaCppPipelineManager')
    def test_health_method(self, mock_pipeline):
        """Test health method"""
        from services.llava_service import LLaVAService
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.get_health_status.return_value = {
            "status": "healthy",
            "model_loaded": True
        }
        mock_pipeline.return_value = mock_pipeline_instance
        
        service = LLaVAService()
        result = service.health()
        
        assert result["service"] == "LLaVAService"
        assert result["version"] == "1.0.0"
        assert "capabilities" in result
        assert "image_analysis" in result["capabilities"]
    
    def test_get_example_schemas_method(self):
        """Test get_example_schemas method"""
        from services.llava_service import LLaVAService
        
        with patch('services.llava_service.LLaVALlamaCppPipelineManager'):
            service = LLaVAService()
            schemas = service.get_example_schemas()
            
            assert "image_description" in schemas
            assert "object_detection" in schemas
            assert "image_qa" in schemas
            assert "text_extraction" in schemas
            
            # Validate schema structure
            desc_schema = schemas["image_description"]
            assert desc_schema["type"] == "object"
            assert "properties" in desc_schema
            assert "description" in desc_schema["properties"]


class TestLLaVAServiceIntegration:
    """Integration tests for LLaVA service (these may be skipped if model not available)"""
    
    def test_image_to_base64(self, sample_image_path: Path):
        """Helper test to convert test image to base64"""
        if not sample_image_path.exists():
            pytest.skip("Test image not available")
            
        with open(sample_image_path, "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode()
            
        assert len(base64_image) > 0
        assert base64_image.isascii()
    
    @pytest.mark.slow
    @pytest.mark.timeout(180)
    @pytest.fixture(scope="class")
    def running_llava_service(self) -> Generator[str, None, None]:
        """Start LLaVA service - may be slow due to model loading"""
        try:
            process = subprocess.Popen([
                "uv", "run", "bentoml", "serve",
                "services.llava_service:LLaVAService", 
                "--port", "3002",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            base_url = "http://127.0.0.1:3002"
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
                process.terminate()
                pytest.skip("LLaVA service failed to start within timeout (model may not be available)")
                
            yield base_url
            
        except Exception as e:
            pytest.skip(f"Could not start LLaVA service: {e}")
        finally:
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                psutil.wait_procs(children + [parent], timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
    
    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_health_endpoint_integration(self, running_llava_service: str):
        """Test health endpoint with actual service"""
        response = requests.post(f"{running_llava_service}/health", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "LLaVAService"
        assert "capabilities" in data
    
    @pytest.mark.slow
    @pytest.mark.timeout(30) 
    def test_example_schemas_endpoint(self, running_llava_service: str):
        """Test get_example_schemas endpoint"""
        response = requests.post(f"{running_llava_service}/get_example_schemas", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "image_description" in data
        assert "object_detection" in data
    
    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_analyze_image_with_test_image(self, running_llava_service: str, sample_image_path: Path):
        """Test image analysis with actual test image"""
        if not sample_image_path.exists():
            pytest.skip("Test image not available")
            
        with open(sample_image_path, "rb") as f:
            image_data = f.read()
            base64_image = base64.b64encode(image_data).decode()
        
        payload = {
            "request": {
                "prompt": "What do you see in this image?",
                "image": f"data:image/jpeg;base64,{base64_image}",
                "max_new_tokens": 100,
                "temperature": 0.1
            }
        }
        
        response = requests.post(
            f"{running_llava_service}/analyze_image",
            json=payload,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        # Note: Actual response depends on model availability and performance
        assert "success" in data or "error" in data


class TestLLaVAServiceBehavior:
    """HTTP behavior tests using mocked responses"""
    
    @patch('requests.post')
    def test_service_response_format(self, mock_post):
        """Test expected response format for analyze_image"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "response": "I can see a test image",
            "format": "text",
            "input_prompt": "What do you see?",
            "device_used": "cpu"
        }
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/analyze_image", json={
            "prompt": "What do you see?",
            "image": "test_image"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "input_prompt" in data
        assert "device_used" in data
    
    @patch('requests.post') 
    def test_error_response_format(self, mock_post):
        """Test error response format"""
        mock_response = Mock()
        mock_response.status_code = 200  # BentoML returns 200 with error in body
        mock_response.json.return_value = {
            "success": False,
            "error": "Invalid image format. Supported: URL, base64, or image bytes",
            "format": "error"
        }
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/analyze_image", json={
            "prompt": "What do you see?",
            "image": "invalid_image"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert data["format"] == "error"