from fastapi import APIRouter, HTTPException, Body, Query, Request
from typing import List, Dict, Any, Optional
from api.models import LLMProvider, LLMStatus, LLMConfigRequest, LLMConfigResponse, LLMConfig
from services.llm_service import LLMService

router = APIRouter()

# Initialize LLM service
llm_service = LLMService()

@router.get("/providers")
async def get_available_providers():
    """Get list of available LLM providers"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI GPT",
                "description": "OpenAI's GPT models (GPT-3.5, GPT-4)",
                "requires_api_key": True,
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Google's Gemini models",
                "requires_api_key": True,
                "models": ["gemini-pro", "gemini-pro-vision"]
            },
            {
                "id": "claude",
                "name": "Anthropic Claude",
                "description": "Anthropic's Claude models",
                "requires_api_key": True,
                "models": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"]
            },
            {
                "id": "local",
                "name": "Local LLM",
                "description": "Local models using llama-cpp",
                "requires_api_key": False,
                "models": ["llama-2-7b", "llama-2-13b", "mistral-7b"]
            }
        ]
    }

@router.post("/configure")
async def configure_provider(config: LLMConfig):
    """Configure an LLM provider with API key"""
    try:
        llm_service.set_api_key(config.provider.value, config.api_key)
        return {
            "message": f"Provider {config.provider.value} configured successfully",
            "provider": config.provider.value
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error configuring provider: {str(e)}"
        )

@router.get("/status")
async def get_llm_status():
    """Get status of LLM providers"""
    status = {
        "openai": {
            "configured": llm_service.openai_client is not None,
            "status": "ready" if llm_service.openai_client else "not_configured"
        },
        "gemini": {
            "configured": llm_service.gemini_client is not None,
            "status": "ready" if llm_service.gemini_client else "not_configured"
        },
        "claude": {
            "configured": llm_service.anthropic_client is not None,
            "status": "ready" if llm_service.anthropic_client else "not_configured"
        },
        "local": {
            "configured": llm_service.local_llm is not None,
            "status": "ready" if llm_service.local_llm else "not_configured"
        }
    }
    
    return status

@router.post("/test")
async def test_provider(provider: str, api_key: str = None):
    """Test if a provider is working correctly"""
    try:
        if api_key:
            llm_service.set_api_key(provider, api_key)
        
        # Simple test question
        test_response = llm_service.ask_question(
            question="Hello, this is a test. Please respond with 'Test successful'.",
            context=["This is a test document."],
            provider=provider
        )
        
        return {
            "provider": provider,
            "status": "success",
            "response": test_response["answer"],
            "confidence": test_response.get("confidence", 0.0)
        }
        
    except Exception as e:
        return {
            "provider": provider,
            "status": "error",
            "error": str(e)
        }

@router.get("/models/{provider}")
async def get_provider_models(provider: str):
    """Get available models for a specific provider"""
    models = {
        "openai": [
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "max_tokens": 4096},
            {"id": "gpt-4", "name": "GPT-4", "max_tokens": 8192},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "max_tokens": 128000}
        ],
        "gemini": [
            {"id": "gemini-pro", "name": "Gemini Pro", "max_tokens": 30720},
            {"id": "gemini-pro-vision", "name": "Gemini Pro Vision", "max_tokens": 30720}
        ],
        "claude": [
            {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "max_tokens": 200000},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "max_tokens": 200000},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "max_tokens": 200000}
        ],
        "local": [
            {"id": "llama-2-7b", "name": "Llama 2 7B", "max_tokens": 4096},
            {"id": "llama-2-13b", "name": "Llama 2 13B", "max_tokens": 4096},
            {"id": "mistral-7b", "name": "Mistral 7B", "max_tokens": 8192}
        ]
    }
    
    if provider not in models:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return {"provider": provider, "models": models[provider]} 