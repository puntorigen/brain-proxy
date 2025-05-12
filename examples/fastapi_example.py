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
from brain_proxy import BrainProxy #, ask, chat, index_file, add_memory
import dotenv, os
# TODO create: ask,chat methods (compatible with langchain), index_file, add_memory (args: tenant, data)
# TODO create method for erasing timespan memory/history with tenant (args: tenant, timespan)

# Load environment variables from .env file
dotenv.load_dotenv()

# Enable debug mode for testing
DEBUG_MODE = True

# Example system prompt to test the new feature
SYSTEM_PROMPT = "You are Claude, a friendly and helpful AI assistant. You are concise, respectful, and you always maintain a warm, conversational tone. You prefer to explain concepts using analogies and examples."

app = FastAPI()

# Define a dummy weather tool
async def get_weather(location: str) -> dict:
    """Dummy weather implementation that always returns sunny and 72°F"""
    print("get_weather called with location:", location)
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "sunny",
        "humidity": "45%"
    }

# Create the tool definition in OpenAI function calling format
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}

# Example: instantiate your BrainProxy class
# Create BrainProxy instance
brain_proxy = BrainProxy(
    tools=[weather_tool],
    # Models in litellm format: "{provider}/{model_name}"
    default_model="openai/gpt-4o-mini",
    memory_model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    # Optional: customize memory settings
    enable_memory=True,
    mem_top_k=6,
    # Add system_prompt parameter to test the new feature
    system_prompt=SYSTEM_PROMPT,
    # Debug mode - will print detailed logs when set to True
    temporal_awareness=True,
    # TODO: external=False, # only allows internal access (ask,chat,etc)
    debug=DEBUG_MODE,
    max_workers=5,
    # Upstash configuration
    upstash_rest_url=os.getenv("UPSTASH_REST_URL"),  # Get this from Upstash dashboard
    upstash_rest_token=os.getenv("UPSTASH_REST_TOKEN"),  # Get this from Upstash dashboard
)

# Add tool implementation to BrainProxy instance
brain_proxy.get_weather = get_weather

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
        },
        "system_prompt": brain_proxy.system_prompt,
        "available_tools": brain_proxy.get_tools_schema()
    }
