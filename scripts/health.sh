#!/bin/bash

# Check health of running BentoML service
# Uses environment variables for configuration

# Source .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Set defaults
BENTOML_HOST=${BENTOML_HOST:-127.0.0.1}
BENTOML_PORT=${BENTOML_PORT:-3000}
BENTOML_PROTOCOL=${BENTOML_PROTOCOL:-http}

curl -X POST ${BENTOML_PROTOCOL}://${BENTOML_HOST}:${BENTOML_PORT}/health \
  -H "Content-Type: application/json" \
  -d '{}'