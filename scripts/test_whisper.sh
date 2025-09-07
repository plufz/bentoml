#!/bin/bash

# Test script for WhisperService
# Tests both URL and file upload transcription endpoints

set -e

# Add UV to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "Testing Whisper Service..."

# Check if service is running
SERVICE_URL="http://127.0.0.1:3000"
echo "Checking if service is running at $SERVICE_URL..."

if ! curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo "❌ Service is not running. Please start it first with:"
    echo "   ./scripts/run_bentoml.sh serve services.whisper_service:WhisperService"
    exit 1
fi

echo "✅ Service is running"

# Test URL transcription endpoint
echo ""
echo "Testing URL transcription..."
RESPONSE=$(curl -s -X POST "$SERVICE_URL/transcribe_url" \
    -H "Content-Type: application/json" \
    -d '{"request": {"url": "https://plufz.com/test-assets/test-english.mp3"}}')

if echo "$RESPONSE" | grep -q "text"; then
    echo "✅ URL transcription successful"
    echo "Response preview: $(echo "$RESPONSE" | head -c 200)..."
else
    echo "❌ URL transcription failed"
    echo "Response: $RESPONSE"
    exit 1
fi

# Test file upload endpoint
echo ""
echo "Testing file upload transcription..."

TEST_AUDIO="test-assets/test-english.mp3"
if [ -f "$TEST_AUDIO" ]; then
    echo "Using test audio file: $TEST_AUDIO"
    RESPONSE=$(curl -s -X POST "$SERVICE_URL/transcribe_file" \
        -F "audio_file=@$TEST_AUDIO")
    
    if echo "$RESPONSE" | grep -q "text"; then
        echo "✅ File upload transcription successful"
        echo "Response preview: $(echo "$RESPONSE" | head -c 200)..."
    else
        echo "❌ File upload transcription failed"
        echo "Response: $RESPONSE"
        exit 1
    fi
else
    echo "⚠️  Test audio file not found: $TEST_AUDIO"
    echo "   Skipping file upload test"
fi

echo ""
echo "🎉 Whisper service testing completed!"