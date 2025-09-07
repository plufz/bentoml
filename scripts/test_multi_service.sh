#!/bin/bash

# Multi-Service BentoML Tester
# Tests all endpoints in the unified multi-service application

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_URL="http://127.0.0.1:3000"
TEST_IMAGE="test-assets/test-english.mp3"
TEST_IMAGE_URL="https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400"

echo -e "${GREEN}🧪 Multi-Service BentoML Tester${NC}"
echo -e "${BLUE}Testing unified service with all AI models...${NC}"
echo ""

# Check if service is running
echo "Checking if multi-service is running at ${SERVICE_URL}..."
if curl -s -f "${SERVICE_URL}" > /dev/null; then
    echo -e "${GREEN}✅ Service is running${NC}"
else
    echo -e "${RED}❌ Service is not running. Please start it first with:${NC}"
    echo "   BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh serve services.multi_service:MultiService"
    exit 1
fi

echo ""

# Test system endpoints
echo -e "${YELLOW}Testing system endpoints...${NC}"

echo "Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/health" -H "Content-Type: application/json" -d '{}')
if echo "$HEALTH_RESPONSE" | grep -q "multi_service_status"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo -e "${BLUE}Health status:${NC}"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null | head -10
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo "Response: $HEALTH_RESPONSE"
fi

echo ""

echo "Testing /info endpoint..."
INFO_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/info" -H "Content-Type: application/json" -d '{}')
if echo "$INFO_RESPONSE" | grep -q "available_services"; then
    echo -e "${GREEN}✅ Info endpoint passed${NC}"
    echo -e "${BLUE}Service info:${NC}"
    echo "$INFO_RESPONSE" | python3 -m json.tool 2>/dev/null
else
    echo -e "${RED}❌ Info endpoint failed${NC}"
    echo "Response: $INFO_RESPONSE"
fi

echo ""

# Test Hello Service
echo -e "${YELLOW}Testing Hello Service...${NC}"

echo "Testing /hello endpoint..."
HELLO_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/hello" \
    -H "Content-Type: application/json" \
    -d '{"request": {"name": "Multi-Service"}}')

if echo "$HELLO_RESPONSE" | grep -q "Hello, Multi-Service!"; then
    echo -e "${GREEN}✅ Hello endpoint passed${NC}"
    echo -e "${BLUE}Response:${NC} $HELLO_RESPONSE"
else
    echo -e "${RED}❌ Hello endpoint failed${NC}"
    echo "Response: $HELLO_RESPONSE"
fi

echo ""

# Test Stable Diffusion Service
echo -e "${YELLOW}Testing Stable Diffusion Service...${NC}"

echo "Testing /generate_image endpoint..."
SD_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/generate_image" \
    -H "Content-Type: application/json" \
    -d '{"request": {"prompt": "a small robot", "num_inference_steps": 1}}' \
    --max-time 60)

if echo "$SD_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✅ Image generation passed${NC}"
    echo -e "${BLUE}Generated image successfully${NC}"
else
    echo -e "${RED}❌ Image generation failed${NC}"
    echo "Response preview: ${SD_RESPONSE:0:200}..."
fi

echo ""

# Test LLaVA Service
echo -e "${YELLOW}Testing LLaVA Service...${NC}"

echo "Testing /example_schemas endpoint..."
SCHEMAS_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/example_schemas" \
    -H "Content-Type: application/json" -d '{}')

if echo "$SCHEMAS_RESPONSE" | grep -q "properties"; then
    echo -e "${GREEN}✅ Example schemas passed${NC}"
    echo -e "${BLUE}Schema available${NC}"
else
    echo -e "${RED}❌ Example schemas failed${NC}"
    echo "Response: $SCHEMAS_RESPONSE"
fi

echo "Testing /analyze_url endpoint..."
ANALYZE_URL_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/analyze_url" \
    -H "Content-Type: application/json" \
    -d "{\"request\": {\"image_url\": \"${TEST_IMAGE_URL}\", \"prompt\": \"Describe this image\", \"mode\": \"raw_text\"}}" \
    --max-time 45)

if echo "$ANALYZE_URL_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✅ URL image analysis passed${NC}"
    echo -e "${BLUE}Image analysis successful${NC}"
else
    echo -e "${RED}❌ URL image analysis failed${NC}"
    echo "Response preview: ${ANALYZE_URL_RESPONSE:0:200}..."
fi

echo ""

# Test Whisper Service
echo -e "${YELLOW}Testing Whisper Service...${NC}"

if [ -f "$TEST_IMAGE" ]; then
    echo "Testing /transcribe_file endpoint with test audio..."
    TRANSCRIBE_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/transcribe_file" \
        -F "audio_file=@${TEST_IMAGE}" \
        --max-time 30)
    
    if echo "$TRANSCRIBE_RESPONSE" | grep -q "success.*true"; then
        echo -e "${GREEN}✅ File transcription passed${NC}"
        echo -e "${BLUE}Transcription successful${NC}"
    else
        echo -e "${RED}❌ File transcription failed${NC}"
        echo "Response preview: ${TRANSCRIBE_RESPONSE:0:200}..."
    fi
else
    echo -e "${YELLOW}⚠️  Test audio file not found, skipping file transcription test${NC}"
fi

echo "Testing /transcribe_url endpoint..."
TRANSCRIBE_URL_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/transcribe_url" \
    -H "Content-Type: application/json" \
    -d '{"request": {"url": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"}}' \
    --max-time 45)

if echo "$TRANSCRIBE_URL_RESPONSE" | grep -q "success.*true"; then
    echo -e "${GREEN}✅ URL transcription passed${NC}"
    echo -e "${BLUE}Transcription successful${NC}"
else
    echo -e "${RED}❌ URL transcription failed${NC}"
    echo "Response preview: ${TRANSCRIBE_URL_RESPONSE:0:200}..."
fi

echo ""

# Summary
echo -e "${GREEN}🎉 Multi-service testing completed!${NC}"
echo ""
echo -e "${BLUE}Available endpoints in the multi-service:${NC}"
echo "• System: /health, /info"
echo "• Hello: /hello" 
echo "• Stable Diffusion: /generate_image"
echo "• LLaVA: /analyze_image, /analyze_structured, /analyze_url, /example_schemas"
echo "• Whisper: /transcribe_file, /transcribe_url"
echo ""
echo -e "${BLUE}Total: 10 endpoints in a single service!${NC}"