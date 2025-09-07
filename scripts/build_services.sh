#!/bin/bash

# Build all BentoML services
BENTOFILE=config/bentofiles/multi-service.yaml ./scripts/run_bentoml.sh build services/multi_service.py