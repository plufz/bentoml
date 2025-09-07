#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONFIG_FILE="scripts/services-config.json"

# Function to show usage
show_usage() {
    echo -e "${BLUE}🏗️  BentoML Services Build Script${NC}"
    echo ""
    echo "Usage: $0 [options] [service_names...]"
    echo ""
    echo "Options:"
    echo "  --list, -l       List available services"
    echo "  --help, -h       Show this help message"
    echo "  --verbose, -v    Show detailed output"
    echo ""
    echo "Examples:"
    echo "  $0                           # Build all services"
    echo "  $0 rag llava                 # Build specific services"
    echo "  $0 --list                    # List available services"
    echo "  $0 --verbose multi-service   # Build with detailed output"
}

# Function to list available services
list_services() {
    echo -e "${BLUE}📋 Available Services:${NC}"
    echo ""
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq not found. Cannot parse service configuration${NC}"
        return 1
    fi
    
    local services_count=$(jq '.services | length' "$CONFIG_FILE" 2>/dev/null)
    
    for (( i=0; i<services_count; i++ )); do
        local name=$(jq -r ".services[$i].name" "$CONFIG_FILE")
        local description=$(jq -r ".services[$i].description // empty" "$CONFIG_FILE") 
        local service_file=$(jq -r ".services[$i].service_file" "$CONFIG_FILE")
        local bentofile=$(jq -r ".services[$i].bentofile" "$CONFIG_FILE")
        local key=$(echo "$service_file" | sed 's/services\///; s/\.py$//' | sed 's/_/-/g')
        
        echo -e "${GREEN}• $name${NC} ${YELLOW}($key)${NC}"
        if [ -n "$description" ]; then
            echo "  $description"
        fi
        echo "  File: $service_file"
        if [ "$bentofile" != "null" ]; then
            echo "  Config: $bentofile"
        fi
        echo ""
    done
}

# Function to build a single service
build_service() {
    local name="$1"
    local service_file="$2" 
    local bentofile="$3"
    local description="$4"
    local verbose="$5"
    
    echo -e "${GREEN}🏗️  Building ${name}...${NC}"
    if [ -n "$description" ]; then
        echo -e "${BLUE}   ${description}${NC}"
    fi
    
    # Check if service file exists
    if [ ! -f "$service_file" ]; then
        echo -e "${RED}❌ Service file not found: $service_file${NC}"
        return 1
    fi
    
    # Build with or without bentofile
    local build_cmd="./scripts/run_bentoml.sh build \"$service_file\""
    
    if [ -n "$bentofile" ] && [ "$bentofile" != "null" ]; then
        if [ ! -f "$bentofile" ]; then
            echo -e "${YELLOW}⚠️  Bentofile not found: $bentofile, using default${NC}"
        else
            build_cmd="BENTOFILE=\"$bentofile\" $build_cmd"
        fi
    fi
    
    if [ "$verbose" = true ]; then
        echo -e "${BLUE}   Command: $build_cmd${NC}"
        eval "$build_cmd"
    else
        eval "$build_cmd" 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${name} built successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to build ${name}${NC}"
        return 1
    fi
}

# Function to find service by key (fuzzy matching)
find_service_by_key() {
    local key="$1"
    local services_count=$(jq '.services | length' "$CONFIG_FILE" 2>/dev/null)
    
    for (( i=0; i<services_count; i++ )); do
        local service_file=$(jq -r ".services[$i].service_file" "$CONFIG_FILE")
        local service_key=$(echo "$service_file" | sed 's/services\///; s/\.py$//' | sed 's/_/-/g')
        local service_name=$(echo "$service_file" | sed 's/services\///; s/_service\.py$//' | sed 's/_/-/g')
        
        if [ "$service_key" = "$key" ] || [ "$service_name" = "$key" ]; then
            echo "$i"
            return 0
        fi
    done
    
    return 1
}

# Parse command line arguments
VERBOSE=false
SPECIFIC_SERVICES=()
SHOW_LIST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --list|-l)
            SHOW_LIST=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        -*)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            SPECIFIC_SERVICES+=("$1")
            shift
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Configuration file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}💡 This file should contain service build configuration${NC}"
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ jq not found. Please install jq to parse service configuration${NC}"
    echo -e "${YELLOW}💡 Install with: brew install jq (macOS) or apt-get install jq (Linux)${NC}"
    exit 1
fi

# Handle list services request
if [ "$SHOW_LIST" = true ]; then
    list_services
    exit 0
fi

echo -e "${GREEN}🚀 Building BentoML Services${NC}"
echo ""

# Parse services from JSON
services_count=$(jq '.services | length' "$CONFIG_FILE" 2>/dev/null)
if [ $? -ne 0 ] || [ "$services_count" = "null" ]; then
    echo -e "${RED}❌ Invalid or corrupted configuration file${NC}"
    exit 1
fi

# Build specific services if requested
if [ ${#SPECIFIC_SERVICES[@]} -gt 0 ]; then
    echo -e "${BLUE}📦 Building specific services: ${SPECIFIC_SERVICES[*]}${NC}"
    echo ""
    
    failed_builds=()
    successful_builds=()
    
    for service_key in "${SPECIFIC_SERVICES[@]}"; do
        service_index=$(find_service_by_key "$service_key")
        
        if [ $? -eq 0 ]; then
            name=$(jq -r ".services[$service_index].name" "$CONFIG_FILE")
            service_file=$(jq -r ".services[$service_index].service_file" "$CONFIG_FILE") 
            bentofile=$(jq -r ".services[$service_index].bentofile" "$CONFIG_FILE")
            description=$(jq -r ".services[$service_index].description // empty" "$CONFIG_FILE")
            
            if build_service "$name" "$service_file" "$bentofile" "$description" "$VERBOSE"; then
                successful_builds+=("$name")
            else
                failed_builds+=("$name")
            fi
        else
            echo -e "${RED}❌ Service not found: $service_key${NC}"
            failed_builds+=("$service_key")
        fi
        echo ""
    done
else
    # Build all services
    echo -e "${BLUE}📦 Building all $services_count services${NC}"
    echo ""
    
    failed_builds=()
    successful_builds=()
    
    for (( i=0; i<services_count; i++ )); do
        name=$(jq -r ".services[$i].name" "$CONFIG_FILE")
        service_file=$(jq -r ".services[$i].service_file" "$CONFIG_FILE") 
        bentofile=$(jq -r ".services[$i].bentofile" "$CONFIG_FILE")
        description=$(jq -r ".services[$i].description // empty" "$CONFIG_FILE")
        
        if build_service "$name" "$service_file" "$bentofile" "$description" "$VERBOSE"; then
            successful_builds+=("$name")
        else
            failed_builds+=("$name")
        fi
        echo ""
    done
fi

# Report results
echo -e "${GREEN}📊 Build Summary${NC}"
echo "Total requested: $((${#successful_builds[@]} + ${#failed_builds[@]}))"
echo -e "${GREEN}Successful: ${#successful_builds[@]}${NC}"
echo -e "${RED}Failed: ${#failed_builds[@]}${NC}"
echo ""

if [ ${#successful_builds[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ Successful builds:${NC}"
    for service in "${successful_builds[@]}"; do
        echo "  • $service"
    done
    echo ""
fi

if [ ${#failed_builds[@]} -gt 0 ]; then
    echo -e "${RED}❌ Failed builds:${NC}"
    for service in "${failed_builds[@]}"; do
        echo "  • $service"
    done
    echo ""
    echo -e "${YELLOW}💡 Use --verbose flag for detailed error output${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 All services built successfully!${NC}"