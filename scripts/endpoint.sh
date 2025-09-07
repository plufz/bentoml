#!/bin/bash

# BentoML Endpoint Testing Script - Configuration-driven version
# Usage: ./scripts/endpoint.sh <endpoint-name> <json-payload>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Default configuration from environment or fallback defaults
DEFAULT_HOST="${BENTOML_HOST:-127.0.0.1}"
DEFAULT_PORT="${BENTOML_PORT:-3000}"
DEFAULT_PROTOCOL="${BENTOML_PROTOCOL:-http}"
BASE_URL="${DEFAULT_PROTOCOL}://${DEFAULT_HOST}:${DEFAULT_PORT}"

CONFIG_FILE="scripts/endpoints-config.json"

# Function to load and display endpoint examples from config
show_examples_from_config() {
    if ! command -v jq &> /dev/null; then
        echo "  $0 health '{}'"
        echo "  $0 hello '{\"name\": \"World\"}'"
        return
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "  $0 health '{}'"
        echo "  $0 hello '{\"name\": \"World\"}'"
        return
    fi
    
    # System endpoints
    local system_count=$(jq '.system_endpoints | length' "$CONFIG_FILE" 2>/dev/null || echo "0")
    for (( i=0; i<system_count && i<2; i++ )); do
        local name=$(jq -r ".system_endpoints[$i].name" "$CONFIG_FILE")
        local payload=$(jq -r ".system_endpoints[$i].example_payload" "$CONFIG_FILE")
        echo "  $0 $name '$payload'"
    done
    
    # Service endpoints (selected examples)
    local services=$(jq -r '.service_endpoints[].endpoints[] | select(.example_payload != "multipart") | "\(.name) \(.example_payload)"' "$CONFIG_FILE" 2>/dev/null | head -6)
    while IFS=' ' read -r name payload; do
        if [ -n "$name" ] && [ -n "$payload" ]; then
            echo "  $0 $name '$payload'"
        fi
    done <<< "$services"
}

# Function to show available endpoints summary
show_available_endpoints_summary() {
    echo "Available Endpoints:"
    
    if ! command -v jq &> /dev/null || [ ! -f "$CONFIG_FILE" ]; then
        echo "  Use --list to see all endpoints (requires jq)"
        return
    fi
    
    # System endpoints count
    local system_count=$(jq '.system_endpoints | length' "$CONFIG_FILE" 2>/dev/null || echo "0")
    echo "  System: $system_count endpoints (health, info, etc.)"
    
    # Service endpoints
    jq -r '.service_endpoints[] | "  \(.service): \(.endpoints | length) endpoints"' "$CONFIG_FILE" 2>/dev/null
    
    echo ""
    local note=$(jq -r '.image_processing_note // empty' "$CONFIG_FILE" 2>/dev/null)
    if [ -n "$note" ]; then
        echo "Note: $note"
    fi
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}🔧 BentoML Endpoint Testing Script${NC}"
    echo ""
    echo "Usage: $0 <endpoint-name> <json-payload> [options]"
    echo ""
    echo "Examples:"
    show_examples_from_config
    echo ""
    echo "Options:"
    echo "  --host <host>     Server host (default: ${DEFAULT_HOST})"
    echo "  --port <port>     Server port (default: ${DEFAULT_PORT})"
    echo "  --verbose         Show detailed curl output"
    echo "  --list, -l        List all available endpoints"
    echo "  --help, -h        Show this help message"
    echo ""
    show_available_endpoints_summary
}

# Function to list all endpoints with details
list_all_endpoints() {
    echo -e "${BLUE}📋 Available BentoML Endpoints${NC}"
    echo ""
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq not found. Cannot parse endpoint configuration${NC}"
        return 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ Configuration file not found: $CONFIG_FILE${NC}"
        return 1
    fi
    
    # System endpoints
    echo -e "${GREEN}🔧 System Endpoints:${NC}"
    local system_count=$(jq '.system_endpoints | length' "$CONFIG_FILE" 2>/dev/null)
    for (( i=0; i<system_count; i++ )); do
        local name=$(jq -r ".system_endpoints[$i].name" "$CONFIG_FILE")
        local description=$(jq -r ".system_endpoints[$i].description" "$CONFIG_FILE")
        local payload=$(jq -r ".system_endpoints[$i].example_payload" "$CONFIG_FILE")
        
        echo -e "  ${YELLOW}$name${NC} - $description"
        echo "    Example: $0 $name '$payload'"
    done
    echo ""
    
    # Service endpoints
    local services_count=$(jq '.service_endpoints | length' "$CONFIG_FILE" 2>/dev/null)
    for (( i=0; i<services_count; i++ )); do
        local service_name=$(jq -r ".service_endpoints[$i].service" "$CONFIG_FILE")
        echo -e "${GREEN}🔧 $service_name:${NC}"
        
        local endpoints_count=$(jq ".service_endpoints[$i].endpoints | length" "$CONFIG_FILE")
        for (( j=0; j<endpoints_count; j++ )); do
            local name=$(jq -r ".service_endpoints[$i].endpoints[$j].name" "$CONFIG_FILE")
            local description=$(jq -r ".service_endpoints[$i].endpoints[$j].description" "$CONFIG_FILE")
            local payload=$(jq -r ".service_endpoints[$i].endpoints[$j].example_payload" "$CONFIG_FILE")
            local requires_file=$(jq -r ".service_endpoints[$i].endpoints[$j].requires_file // false" "$CONFIG_FILE")
            
            echo -e "  ${YELLOW}$name${NC} - $description"
            if [ "$payload" = "multipart" ] || [ "$requires_file" = "true" ]; then
                echo "    Example: curl -X POST http://127.0.0.1:3000/$name -F \"file=@./example.ext\""
            else
                echo "    Example: $0 $name '$payload'"
            fi
        done
        echo ""
    done
    
    # Notes
    local note=$(jq -r '.image_processing_note // empty' "$CONFIG_FILE" 2>/dev/null)
    if [ -n "$note" ]; then
        echo -e "${BLUE}📝 Note:${NC} $note"
    fi
}

# Function to process image responses - extract base64 and save as files
process_image_response() {
    local response="$1"
    local endpoint="$2"
    
    echo -e "${YELLOW}📸 Processing image response...${NC}"
    
    # Create output directory
    if ! mkdir -p endpoint_images 2>/dev/null; then
        echo -e "${RED}❌ Failed to create endpoint_images directory${NC}"
        echo -e "${YELLOW}Response (without image processing):${NC}"
        echo "$response" | python3 -m json.tool
        return
    fi
    
    # Use Python to extract base64 image and save file
    local temp_response_file=$(mktemp)
    echo "$response" > "$temp_response_file"
    
    python3 << EOF
import json
import base64
import sys
from datetime import datetime
import os

try:
    with open('$temp_response_file', 'r') as f:
        response = json.load(f)
    
    # Extract base64 image data based on response structure
    image_base64 = None
    image_format = 'PNG'  # default
    
    if 'image_base64' in response:
        image_base64 = response['image_base64']
        # Get format from response if available
        if 'upscaling_info' in response and 'output_format' in response['upscaling_info']:
            image_format = response['upscaling_info']['output_format']
        elif 'generation_info' in response and 'output_format' in response['generation_info']:
            image_format = response['generation_info']['output_format']
    
    if image_base64:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = image_format.lower()
        if extension == 'jpeg':
            extension = 'jpg'
        
        filename = f'endpoint_images/$endpoint_{timestamp}.{extension}'
        
        # Decode and save image
        image_data = base64.b64decode(image_base64)
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        # Modify response to remove base64 data
        response['image_base64'] = f'<base64 image data extracted to {filename}>'
        
        # Pretty print the modified response
        print(json.dumps(response, indent=2))
        print(f'\\n🎉 Image saved to: {filename}')
        print(f'📊 Image size: {len(image_data)} bytes')
    else:
        # No image data found, just print the response
        print(json.dumps(response, indent=2))
        print('\\n⚠️  No base64 image data found in response')
        
except json.JSONDecodeError:
    print('Error: Invalid JSON response', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error processing image: {e}', file=sys.stderr)
    sys.exit(1)
EOF

    # Clean up temp file
    rm -f "$temp_response_file"
}

# Function to check if endpoint is image endpoint
is_image_endpoint() {
    local endpoint="$1"
    
    if command -v jq &> /dev/null && [ -f "$CONFIG_FILE" ]; then
        # Check if endpoint is marked as image endpoint in config
        local is_image=$(jq -r ".service_endpoints[].endpoints[] | select(.name == \"$endpoint\") | .is_image_endpoint // false" "$CONFIG_FILE" 2>/dev/null)
        if [ "$is_image" = "true" ]; then
            return 0
        fi
    else
        # Fallback to hardcoded list if jq/config unavailable
        local image_endpoints="generate_image upscale_file upscale_url"
        for img_endpoint in $image_endpoints; do
            if [[ "$endpoint" == "$img_endpoint" ]]; then
                return 0
            fi
        done
    fi
    
    return 1
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
        --list|-l)
            list_all_endpoints
            exit 0
            ;;
        --help|-h)
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
BASE_URL="${DEFAULT_PROTOCOL}://${HOST}:${PORT}"

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
        # Check if this is an image-generating endpoint
        if is_image_endpoint "$ENDPOINT"; then
            # Handle image response - extract base64 and save as file
            process_image_response "$RESPONSE" "$ENDPOINT"
        else
            echo -e "${YELLOW}Response:${NC}"
            echo "$RESPONSE" | python3 -m json.tool
        fi
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