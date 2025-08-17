"""
Example usage of the refactored BrainProxy2

This demonstrates the cleaner API and modular architecture
of the new BrainProxy2 implementation.
"""

from fastapi import FastAPI
from brain_proxy.brain_proxy2 import BrainProxy2, BrainProxyConfig

# ==============================================================================
# Basic Usage - Same as before but cleaner
# ==============================================================================

def basic_usage():
    """Basic usage with default settings"""
    proxy = BrainProxy2()
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    # Use with any OpenAI SDK:
    # http://localhost:8000/v1/<tenant>/chat/completions
    return app


# ==============================================================================
# Advanced Configuration - Using the new Config class
# ==============================================================================

def advanced_usage():
    """Advanced usage with custom configuration"""
    
    # Create a configuration object for better organization
    config = BrainProxyConfig(
        # Memory settings
        enable_memory=True,
        memory_model="openai/gpt-4o-mini",
        mem_top_k=10,
        enable_global_memory=True,
        
        # Session settings
        enable_session_memory=True,
        session_ttl_hours=48,
        session_max_messages=200,
        
        # Tool settings
        use_registry_tools=True,
        tool_filtering_model="openai/gpt-3.5-turbo",  # Fast model for filtering
        
        # Model settings
        default_model="openai/gpt-4",
        embedding_model="openai/text-embedding-3-large",
        
        # Storage
        storage_dir="./custom_tenants",
        max_upload_mb=50,
        
        # Features
        temporal_awareness=True,
        system_prompt="You are a helpful AI assistant with long-term memory.",
        debug=True,
        
        # Upstash configuration (optional)
        # upstash_rest_url="https://...",
        # upstash_rest_token="...",
    )
    
    proxy = BrainProxy2(config)
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    return app


# ==============================================================================
# With Custom Callbacks - Cleaner callback management
# ==============================================================================

def with_callbacks():
    """Example with custom callbacks"""
    
    async def auth_hook(request, tenant):
        """Custom authentication"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise Exception("Unauthorized")
        # Validate token...
        return True
    
    async def usage_hook(tenant, tokens, duration):
        """Track usage for billing"""
        print(f"Tenant {tenant}: {tokens} tokens in {duration:.2f}s")
        # Store in database...
    
    async def on_thinking(tenant, state):
        """UI state updates"""
        print(f"Tenant {tenant} is {state}")  # 'thinking' or 'ready'
        # Send websocket update...
    
    async def on_session_end(tenant, session_data):
        """Handle session cleanup"""
        print(f"Session ended for {tenant}")
        print(f"  Messages: {session_data['message_count']}")
        print(f"  Duration: {session_data['created_at']} to {session_data['last_accessed']}")
        # Archive session data...
    
    config = BrainProxyConfig(
        auth_hook=auth_hook,
        usage_hook=usage_hook,
        on_thinking=on_thinking,
        on_session_end=on_session_end,
        debug=True
    )
    
    proxy = BrainProxy2(config)
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    return app


# ==============================================================================
# With Custom Tools - Cleaner tool integration
# ==============================================================================

def with_custom_tools():
    """Example with custom tool handling"""
    
    # Define custom tools in OpenAI format
    custom_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    async def local_tools_handler(tenant, tool_name, args):
        """Handle custom tool execution"""
        if tool_name == "get_weather":
            location = args.get("location", "Unknown")
            # Simulate weather API call
            return f"Weather in {location}: Sunny, 72°F"
        
        raise ValueError(f"Unknown tool: {tool_name}")
    
    config = BrainProxyConfig(
        local_tools_handler=local_tools_handler,
        debug=True
    )
    
    proxy = BrainProxy2(config)
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    # Tools can be set per tenant via API
    # POST /v1/tenant1/tools with custom_tools array
    
    return app


# ==============================================================================
# With Custom Text Extraction - Cleaner document processing
# ==============================================================================

def with_custom_extraction():
    """Example with custom document extraction"""
    from pathlib import Path
    from langchain.schema import Document
    
    def extract_text(path: Path, mime: str):
        """Custom text extraction logic"""
        if mime == "application/pdf":
            # Use PDF library
            import PyPDF2
            # ... extract text from PDF
            return "Extracted PDF content..."
        
        elif mime == "text/markdown":
            # Process markdown specially
            content = path.read_text()
            # Split by headers for better chunking
            sections = content.split("\n## ")
            
            # Return pre-chunked documents
            docs = []
            for i, section in enumerate(sections):
                docs.append(Document(
                    page_content=section,
                    metadata={
                        "source": path.name,
                        "section": i,
                        "type": "markdown"
                    }
                ))
            return docs
        
        else:
            # Default text extraction
            return path.read_text("utf-8", "ignore")
    
    config = BrainProxyConfig(
        extract_text=extract_text,
        debug=True
    )
    
    proxy = BrainProxy2(config)
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    return app


# ==============================================================================
# Direct Service Access - New modular architecture benefit
# ==============================================================================

async def direct_service_usage():
    """
    The new modular architecture allows direct access to services
    for custom integrations or testing
    """
    from brain_proxy.brain_proxy2 import (
        BrainProxy2,
        BrainProxyConfig,
        MemoryService,
        SessionService,
        DocumentService,
        ToolService
    )
    
    config = BrainProxyConfig(debug=True)
    proxy = BrainProxy2(config)
    
    # Direct access to services for custom operations
    
    # 1. Memory Service
    memories = await proxy.memory_service.retrieve(
        tenant="user123",
        query="What did we discuss about Python?",
        session_service=proxy.session_service
    )
    print(f"Found memories: {memories}")
    
    # 2. Session Service
    session = await proxy.session_service.get_or_create("user123:session_abc")
    await session.add_memory("User asked about Python decorators", role="user")
    session_data = session.get_session_data()
    print(f"Session has {session_data['message_count']} messages")
    
    # 3. Document Service
    # Can directly search documents
    docs = await proxy.document_service.search(
        query="machine learning",
        tenant="user123",
        k=5
    )
    print(f"Found {len(docs)} relevant documents")
    
    # 4. Tool Service
    # Can directly filter and execute tools
    tools = proxy.tool_service.get_tools_for_tenant("user123")
    filtered = await proxy.tool_service.filter_tools(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools
    )
    print(f"Filtered to {len(filtered)} relevant tools")
    
    return proxy


# ==============================================================================
# Migration from BrainProxy to BrainProxy2
# ==============================================================================

def migration_example():
    """
    Shows how to migrate from the original BrainProxy to BrainProxy2
    
    The API is mostly compatible, but the configuration is cleaner.
    """
    
    # Old way (BrainProxy):
    # proxy = BrainProxy(
    #     enable_memory=True,
    #     memory_model="openai/gpt-4o-mini",
    #     mem_top_k=6,
    #     enable_global_memory=False,
    #     default_model="openai/gpt-4o-mini",
    #     storage_dir="tenants",
    #     debug=True
    # )
    
    # New way (BrainProxy2) - Option 1: Direct kwargs (compatible)
    proxy = BrainProxy2(
        enable_memory=True,
        memory_model="openai/gpt-4o-mini",
        mem_top_k=6,
        enable_global_memory=False,
        default_model="openai/gpt-4o-mini",
        storage_dir="tenants",
        debug=True
    )
    
    # New way (BrainProxy2) - Option 2: Config object (recommended)
    config = BrainProxyConfig(
        enable_memory=True,
        memory_model="openai/gpt-4o-mini",
        mem_top_k=6,
        enable_global_memory=False,
        default_model="openai/gpt-4o-mini",
        storage_dir="tenants",
        debug=True
    )
    proxy = BrainProxy2(config)
    
    app = FastAPI()
    app.include_router(proxy.router, prefix="/v1")
    
    # The rest of the usage is identical!
    # Same endpoints, same request/response format
    
    return app


# ==============================================================================
# Main entry point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Choose which example to run
    # app = basic_usage()
    # app = advanced_usage()
    # app = with_callbacks()
    # app = with_custom_tools()
    # app = with_custom_extraction()
    app = migration_example()
    
    print("Starting BrainProxy2 server...")
    print("Point your OpenAI SDK to: http://localhost:8000/v1/<tenant>/chat/completions")
    print("Upload files via messages[].content[].file_data")
    print("Set tenant tools via POST /v1/<tenant>/tools")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
