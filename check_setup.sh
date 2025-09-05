#!/bin/bash

# BentoML Setup Checker Script for macOS with UV
# This script verifies the local BentoML setup is working correctly

set -e

# Set PATH to include UV installation locations
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONFIG_FILE="bentoml_config.yaml"

echo -e "${GREEN}🔍 BentoML Setup Verification with UV${NC}"
echo "=================================="

# Check UV installation
echo -n "Checking UV installation... "
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo -e "${GREEN}✅ ${UV_VERSION} found${NC}"
else
    echo -e "${RED}❌ UV not found${NC}"
    echo -e "${YELLOW}   Run ./setup_env.sh to install it${NC}"
    exit 1
fi

# Check pyproject.toml
echo -n "Checking pyproject.toml... "
if [ -f "pyproject.toml" ]; then
    echo -e "${GREEN}✅ Project configuration exists${NC}"
else
    echo -e "${RED}❌ pyproject.toml not found${NC}"
    echo -e "${YELLOW}   Run ./setup_env.sh to create it${NC}"
    exit 1
fi

# Check UV environment and dependencies
echo -n "Checking UV environment... "
if uv sync --dry-run &>/dev/null; then
    echo -e "${GREEN}✅ UV environment is ready${NC}"
else
    echo -e "${RED}❌ UV environment has issues${NC}"
    echo -e "${YELLOW}   Run ./setup_env.sh to fix it${NC}"
    exit 1
fi

# Check BentoML installation
echo -n "Checking BentoML installation... "
if uv run python -c "import bentoml" 2>/dev/null; then
    BENTOML_VERSION=$(uv run python -c "import bentoml; print(bentoml.__version__)")
    echo -e "${GREEN}✅ BentoML ${BENTOML_VERSION} installed${NC}"
else
    echo -e "${RED}❌ BentoML not installed${NC}"
    exit 1
fi

# Check configuration file
echo -n "Checking configuration file... "
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✅ Configuration file exists${NC}"
else
    echo -e "${RED}❌ Configuration file not found${NC}"
    exit 1
fi

# Check required directories
echo -n "Checking storage directories... "
mkdir -p bentos models bentoml_home
echo -e "${GREEN}✅ Storage directories ready${NC}"

# Check dependencies
echo "Checking Python dependencies..."
REQUIRED_PACKAGES=("pandas" "numpy" "fastapi" "uvicorn")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    echo -n "  - $package... "
    if uv run python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✅${NC}"
    else
        echo -e "${RED}❌${NC}"
        MISSING_PACKAGES+=($package)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}   Run ./setup_env.sh to install them${NC}"
fi

# Test BentoML CLI
echo -n "Testing BentoML CLI... "
if uv run bentoml --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ BentoML CLI working${NC}"
else
    echo -e "${RED}❌ BentoML CLI not working${NC}"
fi

# Check ports
echo -n "Checking port 3000 availability... "
if lsof -i :3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 3000 is in use${NC}"
    echo -e "${YELLOW}   You may need to stop the service using port 3000${NC}"
else
    echo -e "${GREEN}✅ Port 3000 available${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}📋 Setup Summary${NC}"
echo "=================================="
echo -e "• UV: ${GREEN}Ready${NC}"
echo -e "• Project Configuration: ${GREEN}Ready${NC}"
echo -e "• BentoML: ${GREEN}Installed${NC}"
echo -e "• Configuration: ${GREEN}Ready${NC}"
echo -e "• Storage: ${GREEN}Ready${NC}"

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Your BentoML setup with UV is ready!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Copy .env.example to .env and customize if needed"
    echo "2. Create a BentoML service (e.g., service.py)" 
    echo "3. Build your service: ./run_bentoml.sh build service.py"
    echo "4. Run your service: ./run_bentoml.sh serve <service_name>:latest"
    echo "5. Visit http://127.0.0.1:3000 in your browser"
    echo ""
    echo -e "${BLUE}UV Commands:${NC}"
    echo "• Run commands: uv run <command>"
    echo "• Add packages: uv add <package>"
    echo "• Update lockfile: uv lock"
    echo "• Sync environment: uv sync"
else
    echo ""
    echo -e "${YELLOW}⚠️  Setup needs attention - run ./setup_env.sh${NC}"
fi