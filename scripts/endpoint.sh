#!/bin/bash

# BentoML Endpoint Testing Script
# Usage: ./scripts/endpoint.sh <endpoint-name> <json-payload>
# Examples:
#   ./scripts/endpoint.sh health '{}'
#   ./scripts/endpoint.sh hello '{"name": "World"}'
#   ./scripts/endpoint.sh generate_image '{"prompt": "A beautiful sunset"}'

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="3000"
BASE_URL="http://${DEFAULT_HOST}:${DEFAULT_PORT}"

# Function to show usage
show_usage() {
    echo -e "${BLUE}🔧 BentoML Endpoint Testing Script${NC}"
    echo ""
    echo "Usage: $0 <endpoint-name> <json-payload> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 health '{}'"
    echo "  $0 hello '{\"name\": \"World\"}'"
    echo "  $0 generate_image '{\"prompt\": \"A beautiful sunset\", \"width\": 512, \"height\": 512}'"
    echo "  $0 analyze_image '{\"image_data\": \"base64...\", \"query\": \"What is in this image?\"}'"
    echo "  $0 transcribe_url '{\"url\": \"https://plufz.com/test-assets/test-english.mp3\"}'"
    echo ""
    echo "Options:"
    echo "  --host <host>     Server host (default: ${DEFAULT_HOST})"
    echo "  --port <port>     Server port (default: ${DEFAULT_PORT})"
    echo "  --verbose         Show detailed curl output"
    echo "  --help           Show this help message"
    echo ""
    echo "Available Endpoints:"
    echo "  System endpoints:"
    echo "    health          - Health check"
    echo "    info            - Service information"
    echo ""
    echo "  Hello Service:"
    echo "    hello           - Simple greeting"
    echo ""
    echo "  Stable Diffusion:"
    echo "    generate_image  - Generate image from text prompt"
    echo ""
    echo "  LLaVA Vision:"
    echo "    analyze_image   - Analyze uploaded image"
    echo "    analyze_structured - Structured image analysis"
    echo "    analyze_url     - Analyze image from URL"
    echo "    example_schemas - Get example JSON schemas"
    echo ""
    echo "  Whisper Audio:"
    echo "    transcribe_file - Transcribe uploaded audio file"
    echo "    transcribe_url  - Transcribe audio from URL"
}

# Parse command line arguments
ENDPOINT=""
PAYLOAD=""
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_PORT}"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$ENDPOINT" ]]; then
                ENDPOINT="$1"
            elif [[ -z "$PAYLOAD" ]]; then
                PAYLOAD="$1"
            else
                echo -e "${RED}❌ Too many arguments${NC}"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [[ -z "$ENDPOINT" ]]; then
    echo -e "${RED}❌ Missing endpoint name${NC}"
    show_usage
    exit 1
fi

if [[ -z "$PAYLOAD" ]]; then
    echo -e "${RED}❌ Missing JSON payload${NC}"
    show_usage
    exit 1
fi

# Update base URL with provided host/port
BASE_URL="http://${HOST}:${PORT}"

# Validate JSON payload
if ! echo "$PAYLOAD" | python3 -m json.tool >/dev/null 2>&1; then
    echo -e "${RED}❌ Invalid JSON payload${NC}"
    echo "Payload: $PAYLOAD"
    exit 1
fi

# Wrap payload in BentoML format (except for system endpoints)
if [[ "$ENDPOINT" == "health" || "$ENDPOINT" == "info" ]]; then
    # System endpoints expect direct payload
    FINAL_PAYLOAD="$PAYLOAD"
else
    # Service endpoints expect {"request": {...}} format
    if [[ "$PAYLOAD" == "{}" ]]; then
        FINAL_PAYLOAD='{"request": {}}'
    else
        # Remove outer braces and wrap in request object
        INNER_PAYLOAD=$(echo "$PAYLOAD" | sed 's/^{//;s/}$//')
        FINAL_PAYLOAD="{\"request\": {$INNER_PAYLOAD}}"
    fi
fi

# Prepare curl command
CURL_CMD="curl -X POST"
CURL_CMD="$CURL_CMD -H 'Content-Type: application/json'"
CURL_CMD="$CURL_CMD -d '$FINAL_PAYLOAD'"

if [[ "$VERBOSE" == true ]]; then
    CURL_CMD="$CURL_CMD -v"
else
    CURL_CMD="$CURL_CMD -s"
fi

CURL_CMD="$CURL_CMD '$BASE_URL/$ENDPOINT'"

# Show request details
echo -e "${BLUE}🚀 Testing BentoML Endpoint${NC}"
echo -e "${YELLOW}Endpoint:${NC} $ENDPOINT"
echo -e "${YELLOW}URL:${NC} $BASE_URL/$ENDPOINT"
echo -e "${YELLOW}Payload:${NC} $FINAL_PAYLOAD"
echo ""

# Execute request
echo -e "${BLUE}📡 Sending request...${NC}"
if [[ "$VERBOSE" == true ]]; then
    echo -e "${YELLOW}Command:${NC} $CURL_CMD"
    echo ""
fi

# Execute and capture response
RESPONSE=$(eval $CURL_CMD)
CURL_EXIT_CODE=$?

echo ""

# Check if request was successful
if [[ $CURL_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ Request successful${NC}"
    
    # Try to format JSON response
    if echo "$RESPONSE" | python3 -m json.tool >/dev/null 2>&1; then
        echo -e "${YELLOW}Response:${NC}"
        echo "$RESPONSE" | python3 -m json.tool
    else
        echo -e "${YELLOW}Response (raw):${NC}"
        echo "$RESPONSE"
    fi
else
    echo -e "${RED}❌ Request failed${NC}"
    echo "Exit code: $CURL_EXIT_CODE"
    
    if [[ -n "$RESPONSE" ]]; then
        echo -e "${YELLOW}Response:${NC}"
        echo "$RESPONSE"
    fi
    
    echo ""
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo "1. Make sure BentoML service is running on $BASE_URL"
    echo "2. Check if the endpoint name is correct"
    echo "3. Verify the JSON payload format"
    echo "4. Use --verbose flag for detailed output"
    echo "5. Try: ./scripts/health.sh to check if service is running"
fi

echo ""