"""
JSON schema validation and parsing utilities for LLaVA services.
Handles structured output validation and response parsing.
"""

import json
import re
from typing import Dict, Any, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError


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


def parse_json_response(
    response_text: str, 
    expected_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse JSON response from LLaVA output, with optional schema validation.
    
    LLaVA responses may contain additional text around the JSON, so this function
    attempts to extract and parse the JSON portion.
    
    Args:
        response_text: Raw response text from LLaVA
        expected_schema: Optional JSON schema to validate against
        
    Returns:
        Dict[str, Any]: Parsed JSON data
        
    Raises:
        ValueError: If JSON parsing or validation fails
    """
    # Try to find JSON in the response
    json_patterns = [
        # Look for JSON blocks between ```json and ```
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
        # Look for JSON objects { ... }
        r'\{.*\}',
        # Look for JSON arrays [ ... ]
        r'\[.*\]'
    ]
    
    json_data = None
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Clean up the match
                clean_match = match.strip()
                json_data = json.loads(clean_match)
                break
            except json.JSONDecodeError:
                continue
        if json_data:
            break
    
    # If no JSON found in patterns, try parsing the entire response
    if json_data is None:
        try:
            json_data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Last resort: try to find JSON-like content and fix common issues
            cleaned_text = response_text.strip()
            # Remove common prefixes/suffixes
            prefixes_to_remove = [
                "Here's the JSON response:",
                "The JSON output is:",
                "JSON:",
                "Response:",
            ]
            for prefix in prefixes_to_remove:
                if cleaned_text.lower().startswith(prefix.lower()):
                    cleaned_text = cleaned_text[len(prefix):].strip()
            
            try:
                json_data = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Unable to parse JSON from response: {str(e)}")
    
    # Validate against schema if provided
    if expected_schema and json_data:
        try:
            validate(instance=json_data, schema=expected_schema)
        except ValidationError as e:
            raise ValueError(f"Response doesn't match expected schema: {str(e)}")
    
    return json_data


def create_json_prompt(
    base_prompt: str, 
    schema: Dict[str, Any], 
    include_example: bool = True
) -> str:
    """
    Create a prompt that instructs LLaVA to respond with structured JSON.
    
    Args:
        base_prompt: The base prompt about the image
        schema: JSON schema for the expected response
        include_example: Whether to include an example in the prompt
        
    Returns:
        str: Formatted prompt for structured JSON output
    """
    schema_str = json.dumps(schema, indent=2)
    
    prompt = f"""{base_prompt}

Please respond with a JSON object that follows this exact schema:

{schema_str}"""

    if include_example and "properties" in schema:
        # Generate a simple example based on the schema
        example = {}
        for prop, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "string":
                example[prop] = f"example_{prop}"
            elif prop_type == "number" or prop_type == "integer":
                example[prop] = 0
            elif prop_type == "boolean":
                example[prop] = True
            elif prop_type == "array":
                example[prop] = []
            elif prop_type == "object":
                example[prop] = {}
        
        example_str = json.dumps(example, indent=2)
        prompt += f"""

Example format:
```json
{example_str}
```"""

    prompt += "\n\nRespond only with valid JSON, no additional text."
    
    return prompt