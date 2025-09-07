#!/bin/bash

# BentoML Local Runner Script for macOS with UV
# This script runs BentoML services locally without Docker using UV

set -e

# Set PATH to include UV installation locations
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config/bentoml.yaml"
ENV_FILE=".env"

echo -e "${GREEN}🚀 Starting BentoML Local Server with UV${NC}"

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ UV not found. Please run ./setup_env.sh first${NC}"
    exit 1
fi

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found. Please run ./setup_env.sh first${NC}"
    exit 1
fi

# Sync dependencies
echo -e "${YELLOW}🔄 Syncing UV environment...${NC}"
uv sync

# Load environment variables if they exist
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}📝 Loading environment variables...${NC}"
    export $(grep -v '^#' $ENV_FILE | xargs)
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating storage directories...${NC}"
mkdir -p bentos models

# Function to start BentoML server
start_server() {
    local bento_tag=$1
    
    if [ -z "$bento_tag" ]; then
        echo -e "${RED}❌ No bento tag specified${NC}"
        echo -e "${BLUE}Usage: $0 [serve|build|list] [bento_tag]${NC}"
        echo -e "${BLUE}Example: $0 serve my_service:latest${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎯 Starting BentoML server for: ${bento_tag}${NC}"
    
    # Source .env file if it exists
    if [ -f .env ]; then
        source .env
    fi
    
    # Start the server with local configuration using UV
    BENTOML_CONFIG_FILE=$CONFIG_FILE uv run bentoml serve $bento_tag \
        --host ${BENTOML_HOST:-127.0.0.1} \
        --port ${BENTOML_PORT:-3000} \
        --reload \
        --development
}

# Function to build a bento
build_bento() {
    local service_file=$1
    
    if [ -z "$service_file" ]; then
        echo -e "${RED}❌ No service file specified${NC}"
        echo -e "${BLUE}Usage: $0 build service.py${NC}"
        exit 1
    fi
    
    if [ ! -f "$service_file" ]; then
        echo -e "${RED}❌ Service file not found: $service_file${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🏗️  Building bento from: ${service_file}${NC}"
    
    # Use custom bentofile if BENTOFILE environment variable is set
    if [ -n "$BENTOFILE" ]; then
        echo -e "${YELLOW}📋 Using custom bentofile: ${BENTOFILE}${NC}"
        uv run bentoml build -f $BENTOFILE $service_file
    else
        uv run bentoml build $service_file
    fi
}

# Function to list bentos
list_bentos() {
    echo -e "${GREEN}📋 Available Bentos:${NC}"
    uv run bentoml list
}

# Function to show help
show_help() {
    echo -e "${GREEN}BentoML Local Runner${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 serve <bento_tag>     - Start BentoML server with specified bento"
    echo "  $0 build <service.py>    - Build a bento from service file"
    echo "  $0 list                  - List available bentos"
    echo "  $0 help                  - Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 serve my_service:latest"
    echo "  $0 build service.py"
    echo "  $0 list"
    echo ""
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  - Config file: $CONFIG_FILE"
    echo "  - Environment file: $ENV_FILE (optional)"
    echo "  - Server URL: ${BENTOML_PROTOCOL:-http}://${BENTOML_HOST:-127.0.0.1}:${BENTOML_PORT:-3000}"
}

# Parse command line arguments
case $1 in
    "serve")
        start_server $2
        ;;
    "build")
        build_bento $2
        ;;
    "list")
        list_bentos
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "")
        echo -e "${YELLOW}⚠️  No command specified${NC}"
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac