#!/bin/bash

# LLaVA Service Tester Script
# This script tests the LLaVA service endpoints

set -e

# Set PATH to include UV installation locations
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVER_URL="http://127.0.0.1:3000"

echo -e "${GREEN}🧪 LLaVA Service Tester${NC}"

# Function to test health endpoint
test_health() {
    echo -e "${YELLOW}Testing health endpoint...${NC}"
    response=$(curl -s -w "%{http_code}" -X POST "$SERVER_URL/health" \
        -H "Content-Type: application/json" \
        -d '{}')
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [[ "$http_code" -ne 200 ]]; then
        echo -e "${RED}❌ Health check failed with HTTP $http_code${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 1
    fi
    
    success=$(echo "$body" | jq -r '.success // empty')
    if [[ "$success" == "false" ]]; then
        echo -e "${RED}❌ Health check returned success: false${NC}"
        echo "$body" | jq '.'
        return 1
    fi
    
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo "$body" | jq '.'
}

# Function to test example schemas endpoint
test_schemas() {
    echo -e "${YELLOW}Testing example schemas endpoint...${NC}"
    response=$(curl -s -w "%{http_code}" -X POST "$SERVER_URL/get_example_schemas" \
        -H "Content-Type: application/json" \
        -d '{}')
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [[ "$http_code" -ne 200 ]]; then
        echo -e "${RED}❌ Schemas test failed with HTTP $http_code${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 1
    fi
    
    success=$(echo "$body" | jq -r '.success // empty')
    if [[ "$success" == "false" ]]; then
        echo -e "${RED}❌ Schemas test returned success: false${NC}"
        echo "$body" | jq '.'
        return 1
    fi
    
    # Check if image_description schema exists
    image_desc=$(echo "$body" | jq -r '.image_description // empty')
    if [[ -z "$image_desc" ]]; then
        echo -e "${RED}❌ No image_description schema found${NC}"
        echo "$body" | jq '.'
        return 1
    fi
    
    echo -e "${GREEN}✅ Schemas test passed${NC}"
    echo "$body" | jq '.image_description'
}

# Function to test image analysis with a simple prompt
test_image_analysis() {
    echo -e "${YELLOW}Testing image analysis with sample image...${NC}"
    
    # Use a simple base64 encoded 32x32 pixel gray PNG for testing
    TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAM0lEQVR4nO3NMQEAMAyEwG+Uv/RKIEs2TgC8trk0p/U4WHCAHCAHyAFygBwgB8gBchDyAb49AcD7t0R0AAAAAElFTkSuQmCC"
    
    response=$(curl -s -w "%{http_code}" -X POST "$SERVER_URL/analyze_image" \
        -H "Content-Type: application/json" \
        -d '{
            "request": {
                "prompt": "What do you see in this image?",
                "image": "'$TEST_IMAGE'",
                "include_raw_response": true
            }
        }')
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [[ "$http_code" -ne 200 ]]; then
        echo -e "${RED}❌ Image analysis failed with HTTP $http_code${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 1
    fi
    
    success=$(echo "$body" | jq -r '.success')
    if [[ "$success" != "true" ]]; then
        echo -e "${RED}❌ Image analysis returned success: $success${NC}"
        error=$(echo "$body" | jq -r '.error // "Unknown error"')
        echo -e "${RED}Error: $error${NC}"
        echo "$body" | jq '.'
        return 1
    fi
    
    echo -e "${GREEN}✅ Image analysis passed${NC}"
    echo "$body" | jq '.success, .format, .device_used'
}

# Function to test structured JSON output
test_json_output() {
    echo -e "${YELLOW}Testing structured JSON output...${NC}"
    
    TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAM0lEQVR4nO3NMQEAMAyEwG+Uv/RKIEs2TgC8trk0p/U4WHCAHCAHyAFygBwgB8gBchDyAb49AcD7t0R0AAAAAElFTkSuQmCC"
    
    response=$(curl -s -w "%{http_code}" -X POST "$SERVER_URL/analyze_image" \
        -H "Content-Type: application/json" \
        -d '{
            "request": {
                "prompt": "Describe this image",
                "image": "'$TEST_IMAGE'",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "objects": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["description"]
                }
            }
        }')
    
    http_code="${response: -3}"
    body="${response%???}"
    
    if [[ "$http_code" -ne 200 ]]; then
        echo -e "${RED}❌ JSON output test failed with HTTP $http_code${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
        return 1
    fi
    
    success=$(echo "$body" | jq -r '.success')
    if [[ "$success" != "true" ]]; then
        echo -e "${RED}❌ JSON output test returned success: $success${NC}"
        error=$(echo "$body" | jq -r '.error // "Unknown error"')
        echo -e "${RED}Error: $error${NC}"
        echo "$body" | jq '.'
        return 1
    fi
    
    echo -e "${GREEN}✅ JSON output test passed${NC}"
    echo "$body" | jq '.success, .format, .response'
}

# Main execution
test_failures=0

case "${1:-all}" in
    "health")
        test_health || test_failures=$((test_failures + 1))
        ;;
    "schemas")
        test_schemas || test_failures=$((test_failures + 1))
        ;;
    "image")
        test_image_analysis || test_failures=$((test_failures + 1))
        ;;
    "json")
        test_json_output || test_failures=$((test_failures + 1))
        ;;
    "all"|*)
        echo -e "${BLUE}Running all tests...${NC}\n"
        test_health || test_failures=$((test_failures + 1))
        echo
        test_schemas || test_failures=$((test_failures + 1))
        echo
        test_image_analysis || test_failures=$((test_failures + 1))
        echo
        test_json_output || test_failures=$((test_failures + 1))
        ;;
esac

echo
if [[ $test_failures -eq 0 ]]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $test_failures test(s) failed!${NC}"
    exit 1
fi