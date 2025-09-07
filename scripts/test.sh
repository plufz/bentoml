#!/bin/bash

# BentoML Test Runner Script
# Runs pytest with UV for BentoML services testing
#
# Usage:
#   ./scripts/test.sh                    # Run all fast tests (unit + behavior)
#   ./scripts/test.sh --all              # Run all tests including slow integration
#   ./scripts/test.sh --coverage         # Run with coverage report
#   ./scripts/test.sh --service example  # Run specific service tests
#   ./scripts/test.sh --help             # Show help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Ensure UV is in PATH
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Adding UV to PATH...${NC}"
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Function to print usage
print_usage() {
    echo -e "${BLUE}BentoML Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)           Run all fast tests (unit + behavior, excludes slow integration)"
    echo "  --all              Run all tests including slow integration tests"
    echo "  --coverage         Run fast tests with coverage report"
    echo "  --coverage-all     Run all tests with coverage report"
    echo "  --service SERVICE  Run tests for specific service (example, llava, stable_diffusion, whisper, upscaler, multi)"
    echo "  --unit             Run only unit tests"
    echo "  --behavior         Run only HTTP behavior tests"
    echo "  --integration      Run only integration tests (slow)"
    echo "  --verbose, -v      Verbose output"
    echo "  --help, -h         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Fast tests only"
    echo "  $0 --all                     # All tests including slow ones"
    echo "  $0 --coverage                # Fast tests with coverage"
    echo "  $0 --service example         # Only example service tests"
    echo "  $0 --unit --verbose          # Unit tests with verbose output"
}

# Parse arguments
PYTEST_ARGS=()
RUN_ALL=false
WITH_COVERAGE=false
VERBOSE=false
SPECIFIC_SERVICE=""
TEST_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --coverage)
            WITH_COVERAGE=true
            shift
            ;;
        --coverage-all)
            WITH_COVERAGE=true
            RUN_ALL=true
            shift
            ;;
        --service)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        --unit)
            TEST_TYPE="Unit"
            shift
            ;;
        --behavior)
            TEST_TYPE="Behavior"
            shift
            ;;
        --integration)
            TEST_TYPE="Integration"
            RUN_ALL=true  # Integration tests are marked as slow
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            PYTEST_ARGS+=("-v")
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Build pytest command
echo -e "${BLUE}🧪 Running BentoML Tests with pytest${NC}"

# Add coverage if requested
if [[ "$WITH_COVERAGE" == true ]]; then
    PYTEST_ARGS+=("--cov=." "--cov-report=term-missing" "--cov-report=html:htmlcov")
    echo -e "${YELLOW}📊 Coverage reporting enabled${NC}"
fi

# Add test selection
if [[ -n "$SPECIFIC_SERVICE" ]]; then
    case "$SPECIFIC_SERVICE" in
        example)
            PYTEST_ARGS+=("tests/test_example_service.py")
            ;;
        llava)
            PYTEST_ARGS+=("tests/test_llava_service.py")
            ;;
        stable_diffusion|sd)
            PYTEST_ARGS+=("tests/test_stable_diffusion_service.py")
            ;;
        whisper)
            PYTEST_ARGS+=("tests/test_whisper_service.py")
            ;;
        upscaler)
            PYTEST_ARGS+=("tests/test_upscaler_service.py")
            ;;
        multi)
            PYTEST_ARGS+=("tests/test_multi_service.py")
            ;;
        *)
            echo -e "${RED}❌ Unknown service: $SPECIFIC_SERVICE${NC}"
            echo "Available services: example, llava, stable_diffusion, whisper, upscaler, multi"
            exit 1
            ;;
    esac
    echo -e "${YELLOW}🎯 Testing specific service: $SPECIFIC_SERVICE${NC}"
elif [[ -n "$TEST_TYPE" ]]; then
    PYTEST_ARGS+=("-k" "$TEST_TYPE")
    echo -e "${YELLOW}🎯 Testing specific type: $TEST_TYPE${NC}"
elif [[ "$RUN_ALL" == false ]]; then
    # Default: exclude slow tests
    PYTEST_ARGS+=("-m" "not slow")
    echo -e "${YELLOW}⚡ Running fast tests only (use --all for integration tests)${NC}"
else
    echo -e "${YELLOW}🔄 Running all tests including slow integration tests${NC}"
fi

# Add short traceback for cleaner output
PYTEST_ARGS+=("--tb=short")

# Run the tests
echo -e "${GREEN}Command: uv run pytest ${PYTEST_ARGS[*]}${NC}"
echo ""

if uv run pytest "${PYTEST_ARGS[@]}"; then
    echo ""
    echo -e "${GREEN}✅ Tests completed successfully!${NC}"
    
    if [[ "$WITH_COVERAGE" == true ]]; then
        echo -e "${BLUE}📊 Coverage report saved to htmlcov/index.html${NC}"
    fi
    
    if [[ "$RUN_ALL" == false ]]; then
        echo -e "${YELLOW}💡 Tip: Run './scripts/test.sh --all' to include slow integration tests${NC}"
    fi
else
    echo ""
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "  - Run './scripts/test.sh --verbose' for more detailed output"
    echo "  - Run './scripts/test.sh --service <name>' to test a specific service"
    echo "  - Check the test output above for specific failure details"
    exit 1
fi