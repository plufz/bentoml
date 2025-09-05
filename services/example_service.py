"""
Example BentoML Service for Testing
This is a simple example service to test your BentoML local setup
"""

import bentoml
from pydantic import BaseModel
from typing import Dict, Any


class HelloRequest(BaseModel):
    name: str = "World"


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10}
)
class HelloService:
    """Simple hello world service for testing BentoML setup"""
    
    @bentoml.api
    def hello(self, request: HelloRequest) -> Dict[str, Any]:
        """Simple hello endpoint that echoes back the input with a greeting"""
        return {
            "message": f"Hello, {request.name}!",
            "status": "success",
            "service": "BentoML Local Test"
        }
    
    @bentoml.api 
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "HelloService",
            "version": "1.0.0"
        }