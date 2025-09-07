# Whisper Service Documentation

🚀 **Audio transcription service** using MLX Whisper for fast, accurate speech-to-text conversion on Apple Silicon.

## Overview

The Whisper Service provides high-quality audio transcription using MLX Whisper Large v3 Turbo, optimized for Apple Silicon devices with MPS acceleration.

## Features

- **🎯 MLX Whisper Large v3 Turbo**: Fast, accurate transcription model
- **📱 Multiple Audio Formats**: MP3, WAV, M4A, FLAC, and more
- **🌐 URL Support**: Transcribe audio files directly from URLs
- **⚡ Apple Silicon Optimized**: MPS acceleration for M1/M2/M3 Macs
- **🔧 Flexible Input**: File uploads or URL-based processing
- **📊 Detailed Response**: Includes transcription metadata and confidence scores

## Quick Start

### Start the Service

```bash
# Individual service
./scripts/start.sh whisper

# Or as part of multi-service (recommended)
./scripts/start.sh
```

### Test the Service

```bash
# Transcribe from URL
./scripts/endpoint.sh transcribe_url '{"url": "https://plufz.com/test-assets/test-english.mp3"}'

# Transcribe from file upload (use curl)
curl -X POST http://127.0.0.1:3000/transcribe_file \
  -F "audio_file=@./test-assets/test-english.mp3"
```

## API Endpoints

### 🌐 `/transcribe_url` - Transcribe from URL

Process an audio file from a URL.

**Request:**
```json
{
  "request": {
    "url": "https://example.com/audio.mp3"
  }
}
```

### 📁 `/transcribe_file` - Transcribe from File Upload

Upload and transcribe an audio file.

**Request (multipart/form-data):**
```bash
curl -X POST http://127.0.0.1:3000/transcribe_file \
  -F "audio_file=@path/to/audio.mp3"
```

## Response Format

```json
{
  "success": true,
  "text": "This is the transcribed text from the audio file.",
  "transcription_info": {
    "duration": 45.2,
    "language": "en",
    "model": "mlx-community/whisper-large-v3-turbo"
  }
}
```

## Supported Audio Formats

**Input:** MP3, WAV, M4A, FLAC, OGG, WEBM, and most audio formats supported by pydub

## Configuration

### Environment Variables

```bash
# Service port (default: 3004)
WHISPER_SERVICE_PORT=3004

# Server configuration
BENTOML_HOST=127.0.0.1
BENTOML_PROTOCOL=http
```

## Testing

Run whisper service tests:

```bash
# Run all whisper tests
./scripts/test.sh --service whisper

# Run specific test classes
uv run pytest tests/test_whisper_service.py
```

## Related Services

- **[Photo Upscaler](photo-upscaler.md)** - Enhance audio thumbnails
- **[LLaVA Vision](llava-service.md)** - Analyze audio spectrograms
- **[Multi-Service](../CLAUDE.md#multi-service-architecture)** - Use all services together

---

📊 **Performance Tip**: MLX Whisper provides excellent performance on Apple Silicon. For Intel Macs, consider using CPU-only inference for consistent results.