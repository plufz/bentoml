"""
HuggingFace environment configuration utilities.
Handles cache directory setup and environment variables.
"""

import os
from pathlib import Path
from typing import Optional


def setup_hf_environment(custom_hf_home: Optional[str] = None) -> str:
    """
    Setup HuggingFace environment variables for model caching.
    
    This function ensures that HF_HOME and related cache variables are properly
    configured, respecting custom cache directories (e.g., external drives).
    
    Args:
        custom_hf_home: Optional custom HF_HOME path. If None, uses environment
                       variable or default cache directory.
                       
    Returns:
        str: The HF_HOME path that was configured
    """
    # Determine HF_HOME path
    if custom_hf_home:
        hf_home = custom_hf_home
    elif 'HF_HOME' in os.environ:
        hf_home = os.environ['HF_HOME']
    else:
        hf_home = os.path.expanduser("~/.cache/huggingface")
    
    # Set HF_HOME
    os.environ['HF_HOME'] = hf_home
    
    # Set related cache variables for consistency
    hub_cache = os.path.join(hf_home, 'hub')
    os.environ['TRANSFORMERS_CACHE'] = hub_cache
    os.environ['HUGGINGFACE_HUB_CACHE'] = hub_cache
    
    # Create directories if they don't exist
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    
    return hf_home


def get_hf_cache_info() -> dict:
    """
    Get information about the current HuggingFace cache configuration.
    
    Returns:
        dict: Cache configuration information
    """
    return {
        'HF_HOME': os.environ.get('HF_HOME', 'Not set'),
        'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE', 'Not set'),
        'HUGGINGFACE_HUB_CACHE': os.environ.get('HUGGINGFACE_HUB_CACHE', 'Not set'),
        'cache_exists': os.path.exists(os.environ.get('HF_HOME', '')),
    }


def print_hf_setup(hf_home: str) -> None:
    """
    Print HuggingFace setup information.
    
    Args:
        hf_home: The HF_HOME path that was configured
    """
    if hf_home == os.path.expanduser("~/.cache/huggingface"):
        print(f"Using default HF_HOME: {hf_home}")
    else:
        print(f"Using custom HF_HOME: {hf_home}")