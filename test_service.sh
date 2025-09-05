#!/bin/bash

# BentoML Service Tester Script for macOS
# This script tests BentoML services locally

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

echo -e "${GREEN}🧪 BentoML Service Tester with UV${NC}"

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ UV not found. Please run ./setup_env.sh first${NC}"
    exit 1
fi

# Sync UV environment
uv sync

# Function to wait for server to be ready
wait_for_server() {
    echo -e "${YELLOW}⏳ Waiting for server to be ready...${NC}"
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "${SERVER_URL}/healthz" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Server is ready!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 1
        ((attempt++))
    done
    
    echo -e "${RED}❌ Server failed to start within ${max_attempts} seconds${NC}"
    return 1
}

# Function to test API endpoints
test_endpoints() {
    echo -e "${BLUE}🔍 Testing API endpoints...${NC}"
    
    # Test health endpoint
    echo -n "Testing health endpoint... "
    HEALTH_RESPONSE=$(curl -s -X POST "${SERVER_URL}/health" \
        -H "Content-Type: application/json" \
        -d '{}' 2>/dev/null || echo "")
    
    if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌${NC}"
        echo -e "${RED}Response: ${HEALTH_RESPONSE}${NC}"
    fi
    
    # Test hello endpoint
    echo -n "Testing hello endpoint... "
    RESPONSE=$(curl -s -X POST "${SERVER_URL}/hello" \
        -H "Content-Type: application/json" \
        -d '{"request": {"name": "BentoML"}}' 2>/dev/null || echo "")
    
    if [[ $RESPONSE == *"Hello, BentoML!"* ]]; then
        echo -e "${GREEN}✅${NC}"
        echo -e "${BLUE}Response: ${RESPONSE}${NC}"
    else
        echo -e "${RED}❌${NC}"
        echo -e "${RED}Response: ${RESPONSE}${NC}"
    fi
    
    # Test OpenAPI docs (BentoML serves docs at root)
    echo -n "Testing web interface... "
    if curl -s -f "${SERVER_URL}/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌${NC}"
    fi
}

# Function to run load test
run_load_test() {
    local requests=${1:-10}
    echo -e "${BLUE}⚡ Running load test with ${requests} requests...${NC}"
    
    for i in $(seq 1 $requests); do
        echo -n "Request $i... "
        RESPONSE=$(curl -s -X POST "${SERVER_URL}/hello" \
            -H "Content-Type: application/json" \
            -d "{\"request\": {\"name\": \"User$i\"}}" 2>/dev/null || echo "")
        
        if [[ $RESPONSE == *"Hello, User$i!"* ]]; then
            echo -e "${GREEN}✅${NC}"
        else
            echo -e "${RED}❌${NC}"
        fi
    done
}

# Function to show service info
show_service_info() {
    echo -e "${BLUE}📊 Service Information${NC}"
    echo "=================================="
    echo "Server URL: ${SERVER_URL}"
    echo "Health Check: ${SERVER_URL}/healthz"
    echo "OpenAPI Docs: ${SERVER_URL}/docs"
    echo "Metrics: ${SERVER_URL}/metrics"
    echo ""
    
    # Show available bentos
    echo -e "${BLUE}Available Bentos:${NC}"
    uv run bentoml list 2>/dev/null || echo "No bentos found"
}

# Main testing function
run_tests() {
    echo -e "${GREEN}🚀 Starting BentoML service tests${NC}"
    
    # Check if server is running
    if ! curl -s -f "${SERVER_URL}/healthz" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Server not running. Starting example service...${NC}"
        
        # Build example service if needed
        if ! uv run bentoml list | grep -q "hello_service"; then
            echo -e "${YELLOW}🏗️  Building example service...${NC}"
            uv run bentoml build example_service.py
        fi
        
        # Start server in background
        echo -e "${YELLOW}🚀 Starting server...${NC}"
        ./run_bentoml.sh serve hello_service:latest &
        SERVER_PID=$!
        
        # Wait for server
        if ! wait_for_server; then
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        
        # Run tests
        test_endpoints
        run_load_test 5
        
        # Clean up
        echo -e "${YELLOW}🧹 Stopping server...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    else
        # Server already running, just test it
        test_endpoints
        run_load_test 5
    fi
    
    echo -e "${GREEN}✅ Tests completed!${NC}"
}

# Parse command line arguments
case $1 in
    "test"|"")
        run_tests
        ;;
    "info")
        show_service_info
        ;;
    "load")
        run_load_test ${2:-10}
        ;;
    "help"|"-h"|"--help")
        echo -e "${GREEN}BentoML Service Tester${NC}"
        echo ""
        echo -e "${YELLOW}Usage:${NC}"
        echo "  $0 test           - Run complete test suite"
        echo "  $0 info           - Show service information"
        echo "  $0 load [N]       - Run load test with N requests (default: 10)"
        echo "  $0 help           - Show this help message"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# No need to deactivate with UV