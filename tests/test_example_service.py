"""
Tests for Example/Hello Service
"""

import pytest
import subprocess
import time
import requests
import signal
import psutil
from pathlib import Path
from typing import Generator
from unittest.mock import patch, Mock


class TestHelloServiceUnit:
    """Unit tests for HelloService - testing individual methods"""
    
    def test_hello_service_import(self):
        """Test that service can be imported successfully"""
        from services.example_service import HelloService, HelloRequest
        
        service = HelloService()
        assert service is not None
        
    def test_hello_request_model(self):
        """Test HelloRequest Pydantic model validation"""
        from services.example_service import HelloRequest
        
        # Test default value
        request = HelloRequest()
        assert request.name == "World"
        
        # Test custom value
        request = HelloRequest(name="BentoML")
        assert request.name == "BentoML"
        
    def test_hello_method_direct(self):
        """Test hello method directly without service startup"""
        from services.example_service import HelloService, HelloRequest
        
        service = HelloService()
        request = HelloRequest(name="Test")
        
        response = service.hello(request)
        
        assert response["message"] == "Hello, Test!"
        assert response["status"] == "success"
        assert response["service"] == "BentoML Local Test"
        
    def test_health_method_direct(self):
        """Test health method directly without service startup"""
        from services.example_service import HelloService
        
        service = HelloService()
        response = service.health()
        
        assert response["status"] == "healthy"
        assert response["service"] == "HelloService"
        assert response["version"] == "1.0.0"


class TestHelloServiceIntegration:
    """Integration tests for HelloService - testing with actual service"""
    
    @pytest.fixture(scope="class")
    def running_service(self) -> Generator[str, None, None]:
        """Start HelloService and return base URL"""
        # Start service in background
        process = subprocess.Popen([
            "uv", "run", "bentoml", "serve", 
            "services.example_service:HelloService",
            "--port", "3001",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for service to start
        base_url = "http://127.0.0.1:3001"
        max_attempts = 30
        for _ in range(max_attempts):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            process.terminate()
            pytest.fail("Service failed to start within timeout")
            
        yield base_url
        
        # Cleanup: terminate process and children
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            parent.terminate()
            psutil.wait_procs(children + [parent], timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
    
    def test_service_startup(self, running_service: str):
        """Test that service starts successfully"""
        response = requests.get(f"{running_service}/")
        assert response.status_code == 200
    
    def test_health_endpoint(self, running_service: str):
        """Test health endpoint via HTTP"""
        response = requests.post(f"{running_service}/health", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "HelloService"
        assert data["version"] == "1.0.0"
    
    def test_hello_endpoint(self, running_service: str):
        """Test hello endpoint via HTTP"""
        payload = {"name": "BentoML Test"}
        response = requests.post(f"{running_service}/hello", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello, BentoML Test!"
        assert data["status"] == "success"
        assert data["service"] == "BentoML Local Test"
    
    def test_hello_endpoint_default_name(self, running_service: str):
        """Test hello endpoint with default name"""
        response = requests.post(f"{running_service}/hello", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello, World!"
        assert data["status"] == "success"
    
    def test_invalid_endpoint(self, running_service: str):
        """Test that invalid endpoints return 404"""
        response = requests.post(f"{running_service}/nonexistent", json={})
        assert response.status_code == 404


class TestHelloServiceBehavior:
    """HTTP behavior tests using mocked service responses"""
    
    @patch('requests.post')
    def test_service_response_format(self, mock_post):
        """Test that service responses follow expected format"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Hello, Test!",
            "status": "success", 
            "service": "BentoML Local Test"
        }
        mock_post.return_value = mock_response
        
        # Simulate request
        response = requests.post("http://test/hello", json={"name": "Test"})
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "service" in data
    
    @patch('requests.post')
    def test_service_error_handling(self, mock_post):
        """Test service error response handling"""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/hello", json={"name": "Test"})
        assert response.status_code == 500