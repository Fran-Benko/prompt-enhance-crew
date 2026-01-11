"""
LLM Loader Utility

This module provides functionality to load and configure Large Language Models (LLMs)
for use with the Prompt Optimizer Crew. It supports Ollama for local models and
can fall back to OpenAI or other providers.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def load_llm(
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    **kwargs
) -> str:
    """
    Load and configure an LLM for use with CrewAI 0.80.0.
    
    CrewAI 0.80.0 uses litellm which supports various providers including Ollama.
    This function returns the appropriate model identifier string for litellm.
    
    Args:
        model_name: Name of the model to load
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation
        provider: LLM provider ('ollama', 'openai', etc.)
        **kwargs: Additional arguments for model configuration
        
    Returns:
        str: Model identifier string for litellm (e.g., 'ollama/llama3.2:3b')
        
    Raises:
        ValueError: If model configuration is invalid
    """
    # Get configuration from environment or use defaults
    provider = provider or os.getenv('LLM_PROVIDER', 'ollama')
    max_tokens = max_tokens or int(os.getenv('DEFAULT_MAX_TOKENS', '512'))
    temperature = temperature or float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
    
    logger.info(f"Configuring LLM with provider: {provider}")
    logger.info(f"Configuration - max_tokens: {max_tokens}, temperature: {temperature}")
    
    if provider.lower() == 'ollama':
        # Ollama configuration
        model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        logger.info(f"Using Ollama model: {model_name}")
        logger.info(f"Ollama base URL: {base_url}")
        
        # For Ollama, litellm expects format: "ollama/model_name"
        # The base URL is configured via environment variable OLLAMA_API_BASE
        os.environ['OLLAMA_API_BASE'] = base_url
        
        return f"ollama/{model_name}"
        
    elif provider.lower() == 'openai':
        # OpenAI configuration
        model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key == 'your_openai_api_key_here':
            raise ValueError("OPENAI_API_KEY not configured in .env file")
        
        logger.info(f"Using OpenAI model: {model_name}")
        
        # For OpenAI, litellm expects just the model name
        return model_name
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_model_info(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about the configured model.
    
    Args:
        provider: LLM provider to get info for
        
    Returns:
        Dict containing model information
    """
    provider = provider or os.getenv('LLM_PROVIDER', 'ollama')
    
    if provider.lower() == 'ollama':
        return {
            'provider': 'ollama',
            'model': os.getenv('OLLAMA_MODEL', 'llama3.2:3b'),
            'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'max_tokens': int(os.getenv('DEFAULT_MAX_TOKENS', '512')),
            'temperature': float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        }
    elif provider.lower() == 'openai':
        return {
            'provider': 'openai',
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            'max_tokens': int(os.getenv('DEFAULT_MAX_TOKENS', '512')),
            'temperature': float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        }
    else:
        return {
            'provider': provider,
            'error': 'Unknown provider'
        }
