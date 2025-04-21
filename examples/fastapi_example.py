"""
A minimal FastAPI example using brain-proxy.
Run with:
    uvicorn examples.fastapi_example:app --reload

Required environment variables:
    - OPENAI_API_KEY (for OpenAI models)
    - AZURE_API_KEY (for Azure models)
    - ANTHROPIC_API_KEY (for Anthropic models)
    etc... (see litellm docs for all supported providers)
"""

from fastapi import FastAPI
# Adjust the import path below if brain_proxy is not installed as a package
from brain_proxy import BrainProxy
import dotenv, os

# Load environment variables from .env file
dotenv.load_dotenv()

# Enable debug mode for testing
DEBUG_MODE = True

app = FastAPI()

# Example: instantiate your BrainProxy class
brain_proxy = BrainProxy(
    # Models in litellm format: "{provider}/{model_name}"
    default_model="openai/gpt-4o-mini",
    memory_model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    # Optional: customize memory settings
    enable_memory=True,
    mem_top_k=6,
    # Debug mode - will print detailed logs when set to True
    debug=DEBUG_MODE,
)

app.include_router(brain_proxy.router, prefix="/v1")    

@app.get("/")
def root():
    # Example usage of brain_proxy; replace with real method as needed
    return {
        "message": "Hello from FastAPI with brain-proxy!",
        "proxy_status": "Ready",
        "debug_mode": "enabled" if DEBUG_MODE else "disabled",
        "models": {
            "default": brain_proxy.default_model,
            "memory": brain_proxy.memory_model,
            "embedding": brain_proxy.embedding_model
        }
    }
