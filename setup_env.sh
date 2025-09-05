#!/bin/bash

# BentoML Local Setup Script for macOS with UV
# This script sets up a Python environment using UV and installs BentoML dependencies

set -e

# Set PATH to include UV installation locations
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up BentoML local environment on macOS with UV${NC}"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}📦 UV not found. Installing UV using the recommended method...${NC}"
    echo -e "${BLUE}ℹ️  Alternative installation methods:${NC}"
    echo -e "${BLUE}   • Homebrew: brew install uv${NC}"
    echo -e "${BLUE}   • pipx: pipx install uv${NC}"
    echo ""
    
    # Install using official installer (recommended method)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # PATH already set at top of script
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ Failed to install UV automatically.${NC}"
        echo -e "${YELLOW}Please install manually using one of these methods:${NC}"
        echo "  1. Homebrew: brew install uv"
        echo "  2. pipx: pipx install uv" 
        echo "  3. Manual: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    echo -e "${GREEN}✅ UV installed successfully! You may need to restart your terminal.${NC}"
fi

UV_VERSION=$(uv --version)
echo -e "${YELLOW}📍 Using ${UV_VERSION}${NC}"

# Check if Python 3.8+ is available
PYTHON_VERSION=$(uv python list | grep -E "python3\.(8|9|10|11|12)" | head -1 | awk '{print $1}' || echo "")
if [ -z "$PYTHON_VERSION" ]; then
    echo -e "${YELLOW}🐍 Installing Python with UV...${NC}"
    uv python install 3.11
    PYTHON_VERSION="python3.11"
fi

echo -e "${YELLOW}📍 Using ${PYTHON_VERSION}${NC}"

# Initialize UV project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}🔧 Initializing UV project...${NC}"
    uv init --no-readme --python "$PYTHON_VERSION"
else
    echo -e "${YELLOW}✅ UV project already initialized${NC}"
fi

# Install dependencies
echo -e "${YELLOW}📦 Installing BentoML and dependencies with UV...${NC}"
uv add "bentoml[io]"
uv add pandas numpy scikit-learn
uv add fastapi uvicorn
uv add prometheus-client
uv add "grpcio" "grpcio-tools"

# Install development dependencies
echo -e "${YELLOW}🛠️  Installing development dependencies...${NC}"
uv add --dev pytest pytest-cov
uv add --dev black isort ruff
uv add --dev jupyter notebook

# Sync dependencies
echo -e "${YELLOW}🔄 Syncing dependencies...${NC}"
uv sync

echo -e "${GREEN}✅ Environment setup complete with UV!${NC}"
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Run commands with: uv run <command>"
echo "2. Run BentoML server: ./run_bentoml.sh"
echo "3. Check configuration: ./check_setup.sh"
echo "4. Activate shell: uv shell (optional)"