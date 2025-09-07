#!/bin/bash

# RAG Service Testing and Management Script
# Provides comprehensive testing and management capabilities for the RAG service

set -e

# Configuration
RAG_HOST="${RAG_HOST:-127.0.0.1}"
RAG_PORT="${RAG_PORT:-3000}"
RAG_BASE_URL="http://${RAG_HOST}:${RAG_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if RAG service is running
check_rag_service() {
    log_info "Checking RAG service health..."
    if curl -s -f "${RAG_BASE_URL}/health" > /dev/null 2>&1; then
        log_success "RAG service is running at ${RAG_BASE_URL}"
        return 0
    else
        log_error "RAG service is not accessible at ${RAG_BASE_URL}"
        return 1
    fi
}

# Function to show RAG service info
show_rag_info() {
    log_info "Getting RAG service information..."
    response=$(curl -s "${RAG_BASE_URL}/get_service_info" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
    else
        log_error "Failed to get RAG service info"
        return 1
    fi
}

# Function to test document ingestion (text)
test_ingest_text() {
    local text="$1"
    local metadata="$2"
    
    if [ -z "$text" ]; then
        text="This is a test document for the RAG system. It contains information about artificial intelligence, machine learning, and natural language processing. RAG systems combine retrieval and generation to provide accurate, context-aware responses."
    fi
    
    if [ -z "$metadata" ]; then
        metadata='{"source": "test", "type": "sample_document"}'
    fi
    
    log_info "Testing text document ingestion..."
    
    payload=$(cat <<EOF
{
    "request": {
        "text": "$text",
        "metadata": $metadata,
        "doc_id": "test-doc-$(date +%s)"
    }
}
EOF
)
    
    response=$(curl -s -X POST "${RAG_BASE_URL}/rag_ingest_text" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
        log_success "Text document ingested successfully"
    else
        log_error "Failed to ingest text document"
        return 1
    fi
}

# Function to test PDF ingestion
test_ingest_pdf() {
    local pdf_path="$1"
    
    if [ ! -f "$pdf_path" ]; then
        log_error "PDF file not found: $pdf_path"
        log_info "Please provide a valid PDF file path"
        return 1
    fi
    
    log_info "Testing PDF document ingestion: $pdf_path"
    
    response=$(curl -s -X POST "${RAG_BASE_URL}/rag_ingest_pdf" \
        -F "pdf_file=@$pdf_path" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
        log_success "PDF document ingested successfully"
    else
        log_error "Failed to ingest PDF document"
        return 1
    fi
}

# Function to test text file ingestion
test_ingest_txt_file() {
    local txt_path="$1"
    
    if [ -z "$txt_path" ]; then
        # Create a temporary test file
        txt_path="/tmp/rag_test_$(date +%s).txt"
        cat > "$txt_path" <<EOF
RAG Testing Document

This is a comprehensive test document for the RAG (Retrieval-Augmented Generation) system.

Key Topics:
1. Natural Language Processing (NLP)
2. Machine Learning and Deep Learning
3. Vector Databases and Embeddings
4. Large Language Models (LLMs)
5. Information Retrieval Systems

The RAG architecture combines the strengths of retrieval-based and generation-based approaches to provide accurate, contextually relevant responses to user queries.

Technical Components:
- Embedding Models: Convert text into vector representations
- Vector Stores: Efficiently store and search document embeddings  
- Language Models: Generate human-like responses based on retrieved context
- Query Processing: Handle user questions and retrieve relevant documents

This document serves as a test case for validating the ingestion, retrieval, and generation capabilities of the RAG system.
EOF
        log_info "Created temporary test file: $txt_path"
    fi
    
    if [ ! -f "$txt_path" ]; then
        log_error "Text file not found: $txt_path"
        return 1
    fi
    
    log_info "Testing text file ingestion: $txt_path"
    
    response=$(curl -s -X POST "${RAG_BASE_URL}/rag_ingest_txt_file" \
        -F "txt_file=@$txt_path" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
        log_success "Text file ingested successfully"
        
        # Cleanup temporary file
        if [[ "$txt_path" == "/tmp/rag_test_"* ]]; then
            rm -f "$txt_path"
            log_info "Cleaned up temporary test file"
        fi
    else
        log_error "Failed to ingest text file"
        return 1
    fi
}

# Function to test RAG query
test_query() {
    local query="$1"
    local max_tokens="$2"
    local temperature="$3"
    local top_k="$4"
    
    if [ -z "$query" ]; then
        query="What is RAG and how does it work?"
    fi
    
    if [ -z "$max_tokens" ]; then
        max_tokens=512
    fi
    
    if [ -z "$temperature" ]; then
        temperature=0.1
    fi
    
    if [ -z "$top_k" ]; then
        top_k=3
    fi
    
    log_info "Testing RAG query: '$query'"
    
    payload=$(cat <<EOF
{
    "request": {
        "query": "$query",
        "max_tokens": $max_tokens,
        "temperature": $temperature,
        "top_k": $top_k,
        "similarity_threshold": 0.7
    }
}
EOF
)
    
    response=$(curl -s -X POST "${RAG_BASE_URL}/rag_query" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$response" | python3 -m json.tool
        log_success "RAG query completed successfully"
    else
        log_error "Failed to execute RAG query"
        return 1
    fi
}

# Function to clear RAG index
clear_index() {
    log_warn "This will clear all documents from the RAG index!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Clearing RAG index..."
        
        response=$(curl -s -X POST "${RAG_BASE_URL}/rag_clear_index" \
            -H "Content-Type: application/json" \
            -d '{}' 2>/dev/null)
        
        if [ $? -eq 0 ]; then
            echo "$response" | python3 -m json.tool
            log_success "RAG index cleared successfully"
        else
            log_error "Failed to clear RAG index"
            return 1
        fi
    else
        log_info "Operation cancelled"
    fi
}

# Function to run complete RAG workflow test
test_complete_workflow() {
    log_info "Running complete RAG workflow test..."
    
    # Check service health
    if ! check_rag_service; then
        log_error "RAG service is not available. Please start it first."
        return 1
    fi
    
    # Ingest test document
    log_info "Step 1: Ingesting test document..."
    if ! test_ingest_text; then
        log_error "Document ingestion failed"
        return 1
    fi
    
    # Wait a moment for indexing
    sleep 2
    
    # Test queries
    log_info "Step 2: Testing various queries..."
    
    local queries=(
        "What is artificial intelligence?"
        "How do RAG systems work?"
        "What are the components of machine learning?"
        "Explain natural language processing"
    )
    
    for query in "${queries[@]}"; do
        log_info "Querying: '$query'"
        if ! test_query "$query" 256 0.1 3; then
            log_warn "Query failed: '$query'"
        fi
        sleep 1
    done
    
    log_success "Complete RAG workflow test finished"
}

# Function to show usage
show_usage() {
    cat <<EOF
RAG Service Testing and Management Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    health                          Check RAG service health
    info                           Show RAG service information
    ingest-text [TEXT] [METADATA]  Ingest text document
    ingest-pdf [PDF_PATH]          Ingest PDF file
    ingest-txt [TXT_PATH]          Ingest text file
    query [QUESTION]               Query the RAG system
    clear                          Clear all documents from index
    workflow                       Run complete workflow test
    help                           Show this help message

EXAMPLES:
    $0 health
    $0 info
    $0 ingest-text "Sample document text" '{"source": "manual"}'
    $0 ingest-pdf ./document.pdf
    $0 ingest-txt ./sample.txt
    $0 query "What is machine learning?"
    $0 clear
    $0 workflow

ENVIRONMENT VARIABLES:
    RAG_HOST                       RAG service host (default: 127.0.0.1)
    RAG_PORT                       RAG service port (default: 3000)

NOTES:
    - Ensure the RAG service is running before using this script
    - Use 'workflow' command for end-to-end testing
    - PDF and text file paths should be accessible from the current directory
EOF
}

# Main script logic
case "${1:-help}" in
    "health")
        check_rag_service
        ;;
    "info")
        show_rag_info
        ;;
    "ingest-text")
        test_ingest_text "$2" "$3"
        ;;
    "ingest-pdf")
        test_ingest_pdf "$2"
        ;;
    "ingest-txt")
        test_ingest_txt_file "$2"
        ;;
    "query")
        test_query "$2" "$3" "$4" "$5"
        ;;
    "clear")
        clear_index
        ;;
    "workflow")
        test_complete_workflow
        ;;
    "help"|*)
        show_usage
        ;;
esac