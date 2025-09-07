#!/bin/bash

# BentoML Test Runner Script - Configuration-driven version
# Usage: ./scripts/test.sh [options] [specific tests...]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration files
TESTS_CONFIG_FILE="scripts/tests-config.json"

# Function to show usage and help
show_usage() {
    echo -e "${BLUE}🧪 BentoML Test Runner Script${NC}"
    echo ""
    echo "Usage: $0 [options] [specific_tests...]"
    echo ""
    echo "Options:"
    echo "  (no args)           Run all fast tests (unit + behavior, excludes slow integration)"
    echo "  --all               Run all tests including slow integration tests"
    echo "  --coverage          Run fast tests with coverage report"
    echo "  --coverage-all      Run all tests with coverage report"
    echo "  --service SERVICE   Run tests for specific service"
    echo "  --unit              Run only unit tests"
    echo "  --behavior          Run only HTTP behavior tests"
    echo "  --integration       Run only integration tests (slow)"
    echo "  --list, -l          List available test services and types"
    echo "  --verbose, -v       Show verbose pytest output"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Fast tests only"
    echo "  $0 --all                     # All tests including slow ones"
    echo "  $0 --coverage                # Fast tests with coverage"
    echo "  $0 --service example         # Only example service tests"
    echo "  $0 --unit --verbose          # Unit tests with verbose output"
    echo ""
    
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        show_available_tests_summary
    else
        echo "Available services: example, llava, stable_diffusion, whisper, upscaler, multi, rag"
        echo "Test types: unit, behavior, integration"
    fi
}

# Function to show available tests summary
show_available_tests_summary() {
    echo "Available Test Services:"
    
    if ! command -v jq &> /dev/null || [ ! -f "$TESTS_CONFIG_FILE" ]; then
        echo "  Use --list to see all services (requires jq)"
        return
    fi
    
    local services_count=$(jq '.test_services | length' "$TESTS_CONFIG_FILE" 2>/dev/null || echo "0")
    local services=$(jq -r '.test_services[] | .name' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
    echo "  Services ($services_count): $services"
    
    local types=$(jq -r '.test_types[] | .name' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
    echo "  Types: $types"
}

# Function to list all available tests
list_all_tests() {
    echo -e "${BLUE}📋 Available BentoML Tests${NC}"
    echo ""
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq not found. Cannot parse test configuration${NC}"
        return 1
    fi
    
    if [ ! -f "$TESTS_CONFIG_FILE" ]; then
        echo -e "${RED}❌ Configuration file not found: $TESTS_CONFIG_FILE${NC}"
        return 1
    fi
    
    # Test services
    echo -e "${GREEN}🧪 Test Services:${NC}"
    local services_count=$(jq '.test_services | length' "$TESTS_CONFIG_FILE" 2>/dev/null)
    for (( i=0; i<services_count; i++ )); do
        local name=$(jq -r ".test_services[$i].name" "$TESTS_CONFIG_FILE")
        local display_name=$(jq -r ".test_services[$i].display_name" "$TESTS_CONFIG_FILE")
        local description=$(jq -r ".test_services[$i].description" "$TESTS_CONFIG_FILE")
        local test_file=$(jq -r ".test_services[$i].test_file" "$TESTS_CONFIG_FILE")
        local aliases=$(jq -r ".test_services[$i].aliases[]?" "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        
        echo -e "  ${YELLOW}$name${NC} - $display_name"
        if [ -n "$aliases" ]; then
            echo "    Aliases: $aliases"
        fi
        echo "    Description: $description"
        echo "    Test file: $test_file"
        echo ""
    done
    
    # Test types
    echo -e "${GREEN}📊 Test Types:${NC}"
    local types_count=$(jq '.test_types | length' "$TESTS_CONFIG_FILE" 2>/dev/null)
    for (( i=0; i<types_count; i++ )); do
        local name=$(jq -r ".test_types[$i].name" "$TESTS_CONFIG_FILE")
        local display_name=$(jq -r ".test_types[$i].display_name" "$TESTS_CONFIG_FILE")
        local description=$(jq -r ".test_types[$i].description" "$TESTS_CONFIG_FILE")
        local marker=$(jq -r ".test_types[$i].pytest_marker" "$TESTS_CONFIG_FILE")
        
        echo -e "  ${YELLOW}$name${NC} - $display_name"
        echo "    Description: $description"
        echo "    Pytest marker: -k '$marker'"
        echo ""
    done
}

# Function to find service by name or alias
find_service_by_name() {
    local search_name="$1"
    
    if ! command -v jq &> /dev/null || [ ! -f "$TESTS_CONFIG_FILE" ]; then
        # Fallback to hardcoded mapping
        case "$search_name" in
            example) echo "tests/test_example_service.py"; return 0 ;;
            llava) echo "tests/test_llava_service.py"; return 0 ;;
            stable_diffusion|sd) echo "tests/test_stable_diffusion_service.py"; return 0 ;;
            whisper) echo "tests/test_whisper_service.py"; return 0 ;;
            upscaler) echo "tests/test_upscaler_service.py"; return 0 ;;
            multi) echo "tests/test_multi_service.py"; return 0 ;;
            rag) echo "tests/test_rag_service.py"; return 0 ;;
            *) return 1 ;;
        esac
    fi
    
    local services_count=$(jq '.test_services | length' "$TESTS_CONFIG_FILE" 2>/dev/null)
    for (( i=0; i<services_count; i++ )); do
        local name=$(jq -r ".test_services[$i].name" "$TESTS_CONFIG_FILE")
        local test_file=$(jq -r ".test_services[$i].test_file" "$TESTS_CONFIG_FILE")
        
        # Check main name
        if [ "$name" = "$search_name" ]; then
            echo "$test_file"
            return 0
        fi
        
        # Check aliases
        local aliases_count=$(jq ".test_services[$i].aliases | length // 0" "$TESTS_CONFIG_FILE" 2>/dev/null)
        for (( j=0; j<aliases_count; j++ )); do
            local alias=$(jq -r ".test_services[$i].aliases[$j]" "$TESTS_CONFIG_FILE")
            if [ "$alias" = "$search_name" ]; then
                echo "$test_file"
                return 0
            fi
        done
    done
    
    return 1
}

# Function to get available service names for error messages
get_available_services() {
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        local services=$(jq -r '.test_services[] | .name' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        local aliases=$(jq -r '.test_services[] | .aliases[]?' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
        if [ -n "$aliases" ]; then
            echo "$services, $aliases"
        else
            echo "$services"
        fi
    else
        echo "example, llava, stable_diffusion, whisper, upscaler, multi, rag"
    fi
}

# Function to build pytest arguments from config
build_pytest_args() {
    local args=()
    
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        # Add short traceback from config
        local short_tb=$(jq -r '.pytest_options.short_traceback[]' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
        if [ -n "$short_tb" ]; then
            args+=($short_tb)
        else
            args+=("--tb=short")
        fi
    else
        args+=("--tb=short")
    fi
    
    echo "${args[@]}"
}

# Parse arguments
PYTEST_ARGS=()
RUN_ALL=false
WITH_COVERAGE=false
VERBOSE=false
SPECIFIC_SERVICE=""
TEST_TYPE=""
SHOW_LIST=false

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
            TEST_TYPE="unit"
            shift
            ;;
        --behavior)
            TEST_TYPE="behavior"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            RUN_ALL=true  # Integration tests are marked as slow
            shift
            ;;
        --list|-l)
            SHOW_LIST=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
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
            # Treat unknown arguments as specific test patterns
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle list request
if [ "$SHOW_LIST" = true ]; then
    list_all_tests
    exit 0
fi

echo -e "${BLUE}🧪 Running BentoML Tests with pytest${NC}"

# Build base pytest arguments
PYTEST_BASE_ARGS=($(build_pytest_args))
PYTEST_ARGS=("${PYTEST_BASE_ARGS[@]}" "${PYTEST_ARGS[@]}")

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v")
fi

# Add coverage if requested
if [[ "$WITH_COVERAGE" == true ]]; then
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        local cov_args=$(jq -r '.pytest_options.coverage_basic[]' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
        if [ -n "$cov_args" ]; then
            PYTEST_ARGS+=($cov_args)
        else
            PYTEST_ARGS+=("--cov=." "--cov-report=term-missing" "--cov-report=html:htmlcov")
        fi
    else
        PYTEST_ARGS+=("--cov=." "--cov-report=term-missing" "--cov-report=html:htmlcov")
    fi
    echo -e "${YELLOW}📊 Coverage reporting enabled${NC}"
fi

# Add test selection
if [[ -n "$SPECIFIC_SERVICE" ]]; then
    service_test_file=$(find_service_by_name "$SPECIFIC_SERVICE")
    if [ $? -eq 0 ]; then
        PYTEST_ARGS+=("$service_test_file")
        echo -e "${YELLOW}🎯 Testing specific service: $SPECIFIC_SERVICE${NC}"
    else
        echo -e "${RED}❌ Unknown service: $SPECIFIC_SERVICE${NC}"
        echo "Available services: $(get_available_services)"
        exit 1
    fi
elif [[ -n "$TEST_TYPE" ]]; then
    # Find test type marker from config
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        local marker=$(jq -r ".test_types[] | select(.name == \"$TEST_TYPE\") | .pytest_marker" "$TESTS_CONFIG_FILE" 2>/dev/null)
        if [ -n "$marker" ] && [ "$marker" != "null" ]; then
            PYTEST_ARGS+=("-k" "$marker")
            echo -e "${YELLOW}🎯 Testing specific type: $TEST_TYPE ($marker)${NC}"
        else
            echo -e "${RED}❌ Unknown test type: $TEST_TYPE${NC}"
            echo "Available types: unit, behavior, integration"
            exit 1
        fi
    else
        # Fallback mapping
        case "$TEST_TYPE" in
            unit) PYTEST_ARGS+=("-k" "Unit") ;;
            behavior) PYTEST_ARGS+=("-k" "Behavior") ;;
            integration) PYTEST_ARGS+=("-k" "Integration") ;;
            *)
                echo -e "${RED}❌ Unknown test type: $TEST_TYPE${NC}"
                echo "Available types: unit, behavior, integration"
                exit 1
                ;;
        esac
        echo -e "${YELLOW}🎯 Testing specific type: $TEST_TYPE${NC}"
    fi
elif [[ "$RUN_ALL" == false ]]; then
    # Default: exclude slow tests
    if command -v jq &> /dev/null && [ -f "$TESTS_CONFIG_FILE" ]; then
        local excludes=$(jq -r '.default_excludes[]' "$TESTS_CONFIG_FILE" 2>/dev/null | tr '\n' ' ')
        if [ -n "$excludes" ]; then
            PYTEST_ARGS+=($excludes)
        else
            PYTEST_ARGS+=("-m" "not slow")
        fi
    else
        PYTEST_ARGS+=("-m" "not slow")
    fi
    echo -e "${YELLOW}⚡ Running fast tests only (use --all for integration tests)${NC}"
else
    echo -e "${YELLOW}🔄 Running all tests including slow integration tests${NC}"
fi

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
        echo -e "${YELLOW}💡 Run with --all to include slow integration tests${NC}"
    fi
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${YELLOW}💡 Use --verbose for detailed output${NC}"
    exit 1
fi