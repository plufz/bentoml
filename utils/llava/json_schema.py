"""
JSON schema validation utilities for LLaVA services.
Handles JSON schema validation for structured output.
"""

from typing import Dict, Any
import jsonschema


def validate_json_schema(schema: Dict[str, Any]) -> bool:
    """
    Validate if a dictionary is a valid JSON schema.
    
    Args:
        schema: Dictionary representing JSON schema
        
    Returns:
        bool: True if valid JSON schema
    """
    try:
        # Check if it's a valid JSON schema by trying to use it
        jsonschema.Draft7Validator.check_schema(schema)
        return True
    except Exception:
        return False


