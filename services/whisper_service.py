import os
import tempfile
import requests
import mimetypes
import orjson
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from typing import Dict, Any, Annotated
from pydantic import BaseModel, HttpUrl
from pathlib import Path
import bentoml
import mlx_whisper


class TranscribeUrlRequest(BaseModel):
    url: HttpUrl


class TranscribeResponse(BaseModel):
    text: str
    segments: list
    language: str


@bentoml.service()
class WhisperService:
    """
    BentoML service for audio transcription using MLX Whisper.
    Supports transcription from URLs and uploaded files.
    """
    
    def __init__(self):
        self.model_name = "mlx-community/whisper-large-v3-turbo"
    
    @bentoml.api
    def transcribe_url(self, request: TranscribeUrlRequest) -> Dict[str, Any]:
        """
        Transcribe audio from a URL.
        
        Args:
            request: Request containing the audio URL
            
        Returns:
            Transcription result as JSON
        """
        try:
            # Download audio from URL
            tmp_path = self._download_url(request.url)
            
            # Transcribe the audio file
            result = self._transcribe_file(tmp_path)
            
            # Clean up temp file
            os.remove(tmp_path)
            
            # Convert to JSON-safe format
            safe_json_string = orjson.dumps(result)
            safe_data = orjson.loads(safe_json_string)
            
            return {
                "success": True,
                "transcription": safe_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": []
            }
    
    @bentoml.api
    def transcribe_file(
        self, 
        audio_file: Path
    ) -> Dict[str, Any]:
        """
        Transcribe an uploaded audio file.
        
        Args:
            audio_file: Audio file path (automatically handled by BentoML)
            
        Returns:
            Transcription result as JSON
        """
        try:
            # File is already saved by BentoML, just transcribe it
            result = self._transcribe_file(str(audio_file))
            
            # Convert to JSON-safe format
            safe_json_string = orjson.dumps(result)
            safe_data = orjson.loads(safe_json_string)
            
            return {
                "success": True,
                "transcription": safe_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": []
            }
    
    def _download_url(self, url: HttpUrl) -> str:
        """
        Download audio file from URL to temporary file.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to temporary file
        """
        # Download the audio file
        response = requests.get(str(url), stream=True, timeout=15)
        response.raise_for_status()
        
        # Try to guess file extension from Content-Type or URL
        content_type = response.headers.get("Content-Type", "")
        extension = mimetypes.guess_extension(content_type)
        
        if not extension:
            # Fallback to file extension in URL
            parsed = urlparse(str(url))
            path = parsed.path
            _, ext = os.path.splitext(path)
            extension = ext if ext else ".mp3"  # Default fallback
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            # Handle both real responses and mock responses in tests
            if hasattr(response, 'iter_content') and callable(response.iter_content):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        tmp.write(chunk)
            else:
                # Fallback for mocked responses
                tmp.write(response.content)
            tmp_path = tmp.name
        
        return tmp_path
    
    def _convert_wav(self, file_path: str) -> str:
        """
        Convert audio file to WAV format (mono, 16kHz).
        
        Args:
            file_path: Path to input audio file
            
        Returns:
            Path to converted WAV file
        """
        # Convert to WAV (mono, 16kHz) using pydub
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        temp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        
        return temp_wav.name
    
    def _transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file using MLX Whisper.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcription result
        """
        # Convert to WAV format
        wav_filepath = self._convert_wav(file_path)
        
        try:
            # Transcribe with mlx-whisper
            result = mlx_whisper.transcribe(wav_filepath, path_or_hf_repo=self.model_name)
            
            return result
            
        finally:
            # Clean up temp WAV file
            if os.path.exists(wav_filepath):
                os.remove(wav_filepath)
    
    @bentoml.api
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "service": "WhisperService",
            "version": "1.0.0",
            "model": self.model_name,
            "capabilities": ["url_transcription", "file_transcription"],
            "status": "healthy"
        }