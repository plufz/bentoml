#!/bin/bash

# Check health of running BentoML service
curl -X POST http://127.0.0.1:3000/health \
  -H "Content-Type: application/json" \
  -d '{}'