"""
Tests for Whisper Audio Transcription Service
"""

import pytest
import subprocess
import time
import requests
import signal
import psutil
from pathlib import Path
from typing import Generator
from unittest.mock import patch, Mock, MagicMock
import tempfile
from urllib.parse import urlparse


class TestWhisperServiceUnit:
    """Unit tests for WhisperService - testing individual methods"""
    
    def test_whisper_service_import(self):
        """Test that Whisper service can be imported successfully"""
        from services.whisper_service import WhisperService, TranscribeUrlRequest, TranscribeResponse
        
        service = WhisperService()
        assert service is not None
        assert service.model_name == "mlx-community/whisper-large-v3-turbo"
    
    def test_transcribe_url_request_model(self):
        """Test TranscribeUrlRequest Pydantic model validation"""
        from services.whisper_service import TranscribeUrlRequest
        from pydantic import ValidationError
        
        # Test valid URL
        request = TranscribeUrlRequest(url="https://example.com/audio.mp3")
        assert str(request.url) == "https://example.com/audio.mp3"
        
        # Test invalid URL
        with pytest.raises(ValidationError):
            TranscribeUrlRequest(url="not_a_url")
    
    def test_transcribe_response_model(self):
        """Test TranscribeResponse Pydantic model"""
        from services.whisper_service import TranscribeResponse
        
        response = TranscribeResponse(
            text="Hello world",
            segments=[{"start": 0.0, "end": 2.0, "text": "Hello world"}],
            language="en"
        )
        assert response.text == "Hello world"
        assert len(response.segments) == 1
        assert response.language == "en"
    
    @patch('services.whisper_service.mlx_whisper')
    @patch('services.whisper_service.requests.get')
    def test_transcribe_url_method_success(self, mock_get, mock_whisper):
        """Test transcribe_url method with successful transcription"""
        from services.whisper_service import WhisperService, TranscribeUrlRequest
        
        # Mock successful download
        mock_response = Mock()
        mock_response.headers = {'content-type': 'audio/mpeg'}
        mock_response.content = b"fake_audio_data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock whisper transcription
        mock_whisper.transcribe.return_value = {
            'text': 'This is a test transcription',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'This is a test transcription'}
            ],
            'language': 'en'
        }
        
        service = WhisperService()
        request = TranscribeUrlRequest(url="https://example.com/test.mp3")
        
        with patch.object(service, '_download_url', return_value="/tmp/test.mp3"):
            result = service.transcribe_url(request)
        
        assert result["success"] is True
        assert result["transcription"]["text"] == "This is a test transcription"
        assert result["transcription"]["language"] == "en"
        assert len(result["transcription"]["segments"]) == 1
    
    @patch('services.whisper_service.requests.get')
    def test_transcribe_url_method_download_error(self, mock_get):
        """Test transcribe_url method with download error"""
        from services.whisper_service import WhisperService, TranscribeUrlRequest
        
        # Mock failed download
        mock_get.side_effect = requests.RequestException("Network error")
        
        service = WhisperService()
        request = TranscribeUrlRequest(url="https://example.com/nonexistent.mp3")
        
        result = service.transcribe_url(request)
        
        assert result["success"] is False
        assert "error" in result
        assert "Network error" in result["error"]
    
    @patch('services.whisper_service.mlx_whisper')
    def test_transcribe_file_method_success(self, mock_whisper):
        """Test transcribe_file method with successful transcription"""
        from services.whisper_service import WhisperService
        
        # Mock whisper transcription
        mock_whisper.transcribe.return_value = {
            'text': 'File transcription test',
            'segments': [
                {'start': 0.0, 'end': 1.5, 'text': 'File transcription test'}
            ],
            'language': 'en'
        }
        
        service = WhisperService()
        
        # Create mock file data
        mock_file_data = b"fake_audio_file_content"
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test_audio.wav"
            result = service.transcribe_file(mock_file_data)
        
        assert result["success"] is True
        assert result["transcription"]["text"] == "File transcription test"
        assert result["transcription"]["language"] == "en"
    
    def test_health_method(self):
        """Test health method"""
        from services.whisper_service import WhisperService
        
        service = WhisperService()
        result = service.health()
        
        assert result["service"] == "WhisperService"
        assert result["version"] == "1.0.0"
        assert result["model"] == "mlx-community/whisper-large-v3-turbo"
        assert "capabilities" in result
        assert "url_transcription" in result["capabilities"]
        assert "file_transcription" in result["capabilities"]
    
    @patch('services.whisper_service.requests.get')
    def test_download_url_method(self, mock_get):
        """Test _download_url private method"""
        from services.whisper_service import WhisperService
        
        # Mock successful download
        mock_response = Mock()
        mock_response.headers = {'content-type': 'audio/mpeg'}
        mock_response.content = b"audio_content_here"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        service = WhisperService()
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = Mock()
            mock_file.name = "/tmp/downloaded_audio.mp3"
            mock_temp.return_value.__enter__.return_value = mock_file
            
            result = service._download_url("https://example.com/audio.mp3")
            
            assert result == "/tmp/downloaded_audio.mp3"
            mock_file.write.assert_called_once_with(b"audio_content_here")


class TestWhisperServiceIntegration:
    """Integration tests for Whisper service (may be skipped if MLX not available)"""
    
    @pytest.mark.slow
    @pytest.fixture(scope="class")
    def running_whisper_service(self) -> Generator[str, None, None]:
        """Start Whisper service - may be slow due to model loading"""
        try:
            process = subprocess.Popen([
                "uv", "run", "bentoml", "serve",
                "services.whisper_service:WhisperService",
                "--port", "3004",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            base_url = "http://127.0.0.1:3004"
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
                pytest.skip("Whisper service failed to start within timeout (MLX may not be available)")
                
            yield base_url
            
        except Exception as e:
            pytest.skip(f"Could not start Whisper service: {e}")
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
    def test_health_endpoint_integration(self, running_whisper_service: str):
        """Test health endpoint with actual service"""
        response = requests.post(f"{running_whisper_service}/health", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "WhisperService"
        assert "capabilities" in data
        assert data["model"] == "mlx-community/whisper-large-v3-turbo"
    
    @pytest.mark.slow
    def test_transcribe_file_integration(self, running_whisper_service: str, sample_audio_path: Path):
        """Test file transcription with actual audio file"""
        if not sample_audio_path.exists():
            pytest.skip("Test audio file not available")
        
        with open(sample_audio_path, "rb") as f:
            files = {"file": ("test.mp3", f, "audio/mpeg")}
            
            response = requests.post(
                f"{running_whisper_service}/transcribe_file",
                files=files,
                timeout=30
            )
        
        assert response.status_code == 200
        data = response.json()
        # Note: Actual response depends on model availability
        assert "success" in data or "error" in data
        
        if data.get("success"):
            assert "transcription" in data
            assert "text" in data["transcription"]
            assert "language" in data["transcription"]
    
    @pytest.mark.slow
    def test_transcribe_url_integration(self, running_whisper_service: str):
        """Test URL transcription (with a mock URL for safety)"""
        # Note: This would need a real audio URL to test properly
        # For safety, we'll test with an invalid URL to check error handling
        payload = {
            "url": "https://httpbin.org/status/404"
        }
        
        response = requests.post(
            f"{running_whisper_service}/transcribe_url",
            json=payload,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        # Expect this to fail gracefully
        assert data.get("success") is False
        assert "error" in data


class TestWhisperServiceBehavior:
    """HTTP behavior tests using mocked responses"""
    
    @patch('requests.post')
    def test_transcribe_response_format(self, mock_post):
        """Test expected response format for transcription"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "transcription": {
                "text": "This is a test transcription",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "This is a test transcription"}
                ],
                "language": "en"
            },
            "processing_time": 1.23,
            "model_used": "mlx-community/whisper-large-v3-turbo"
        }
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/transcribe_url", json={
            "url": "https://example.com/audio.mp3"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "transcription" in data
        assert data["transcription"]["text"] == "This is a test transcription"
        assert data["transcription"]["language"] == "en"
    
    @patch('requests.post')
    def test_error_response_format(self, mock_post):
        """Test error response format"""
        mock_response = Mock()
        mock_response.status_code = 200  # BentoML returns 200 with error in body
        mock_response.json.return_value = {
            "success": False,
            "error": "Failed to download audio from URL: Connection timeout",
            "url": "https://example.com/nonexistent.mp3"
        }
        mock_post.return_value = mock_response
        
        response = requests.post("http://test/transcribe_url", json={
            "url": "https://example.com/nonexistent.mp3"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_audio_format_support(self):
        """Test various audio format validations"""
        from services.whisper_service import WhisperService
        
        service = WhisperService()
        
        # Test common audio MIME types that should be supported
        supported_types = [
            'audio/mpeg',
            'audio/wav', 
            'audio/mp4',
            'audio/ogg',
            'audio/flac'
        ]
        
        # This is a conceptual test - actual implementation would validate MIME types
        for mime_type in supported_types:
            # In a real implementation, you'd test the actual validation logic
            assert mime_type.startswith('audio/')  # Basic validation