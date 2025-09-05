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
    curl -X POST "$SERVER_URL/health" \
        -H "Content-Type: application/json" \
        -d '{}' \
        | jq '.'
}

# Function to test example schemas endpoint
test_schemas() {
    echo -e "${YELLOW}Testing example schemas endpoint...${NC}"
    curl -X POST "$SERVER_URL/get_example_schemas" \
        -H "Content-Type: application/json" \
        -d '{}' \
        | jq '.image_description'
}

# Function to test image analysis with a simple prompt
test_image_analysis() {
    echo -e "${YELLOW}Testing image analysis with sample image...${NC}"
    
    # Use a simple base64 encoded 1x1 pixel PNG for testing
    TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9hVgWfwAAAABJRU5ErkJggg=="
    
    curl -X POST "$SERVER_URL/analyze_image" \
        -H "Content-Type: application/json" \
        -d '{
            "request": {
                "prompt": "What do you see in this image?",
                "image": "'$TEST_IMAGE'",
                "include_raw_response": true
            }
        }' \
        | jq '.success, .format, .device_used'
}

# Function to test structured JSON output
test_json_output() {
    echo -e "${YELLOW}Testing structured JSON output...${NC}"
    
    TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9hVgWfwAAAABJRU5ErkJggg=="
    
    curl -X POST "$SERVER_URL/analyze_image" \
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
        }' \
        | jq '.success, .format, .response'
}

# Main execution
case "${1:-all}" in
    "health")
        test_health
        ;;
    "schemas")
        test_schemas
        ;;
    "image")
        test_image_analysis
        ;;
    "json")
        test_json_output
        ;;
    "all"|*)
        echo -e "${BLUE}Running all tests...${NC}\n"
        test_health
        echo
        test_schemas
        echo
        test_image_analysis
        echo
        test_json_output
        ;;
esac

echo -e "${GREEN}✅ Testing complete!${NC}"