# 🧠 brain-proxy

**Turn any FastAPI backend into a fully featured OpenAI-compatible LLM proxy — with memory, RAG, streaming, and file uploads.**

> Like the OpenAI `/chat/completions` endpoint — but with context, memory, and smart file ingestion.

---

## ✨ Features

- ✅ OpenAI-compatible `/chat/completions` (drop-in SDK support)
- ✅ Multi-tenant routing (`/v1/<tenant>/chat/completions`)
- ✅ File ingestion via `file_data` messages
- ✅ RAG with Chroma + LangChain
- ✅ LangMem-powered long & short-term memory
- ✅ Tenant-specific file storage for improved organization and isolation
- ✅ Streaming via Server-Sent Events
- ✅ Custom text extractor support for PDFs, CSVs, etc.
- ✅ Real-time processing feedback via `on_thinking` callback
- ✅ No frontend changes required
- ✅ **Now uses LiteLLM by default — specify any model using `provider/model` (e.g., `openai/gpt-4o`, `cerebras/llama3-70b-instruct`)**
- ✅ **NEW: Ephemeral Session Memory** — Separate persistent tenant knowledge from temporary user sessions

---

## 🆕 Ephemeral Session Memory

Brain-proxy now supports **ephemeral session memory**, perfect for customer support, chat applications, and multi-user scenarios where you need:
- **Persistent tenant knowledge** (company info, policies, products) 
- **Temporary session context** (individual user conversations)

### How It Works

Use a colon `:` separator in your tenant ID to create a session:

```python
# Base tenant only (persistent memory)
/v1/acme/chat/completions

# Tenant with session (persistent + ephemeral)
/v1/acme:+15551234567/chat/completions      # Phone support
/v1/acme:user@email.com/chat/completions     # Email support
/v1/acme:chat_session_123/chat/completions   # Web chat
```

### Key Features

- **Session Persistence**: Sessions remain active within TTL (default 24 hours)
- **Memory Overflow Protection**: Automatic summarization prevents unbounded growth
- **File Upload Blocking**: Sessions cannot upload files (security feature)
- **Session Callbacks**: Extract insights when sessions end
- **Intelligent Memory Retrieval**: Combines base knowledge with session context

### Example Usage

```python
from brain_proxy import BrainProxy

async def on_session_end(tenant_id: str, session_data: dict):
    """Called when a session expires."""
    print(f"Session {tenant_id} ended with {session_data['message_count']} messages")
    # Extract valuable insights, store feedback, etc.

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    enable_session_memory=True,  # Enable ephemeral sessions
    session_ttl_hours=24,         # Session lifetime
    session_max_messages=100,     # Max messages before summarization
    on_session_end=on_session_end # Callback for session cleanup
)
```

### Customer Support Example

```python
import openai

# Configure for your tenant with session
openai.api_base = "http://localhost:8000/v1/support:+15551234567"

# First interaction - creates new session
response1 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "I need help with my order #12345"}]
)

# Later in conversation - session context preserved
response2 = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": "What about the issue I mentioned?"}]
)
# The AI remembers the order number from earlier!
```

See [examples/ephemeral_session_example.py](examples/ephemeral_session_example.py) for a complete demonstration.

---

## 🚀 Installation

```bash
pip install brain-proxy
```

---

## ⚡ Quickstart

```python
from fastapi import FastAPI
from brain_proxy import BrainProxy

# Optional: Add callback for UI feedback
def on_thinking(tenant_id: str, state: str):
    if state == 'thinking':
        print(f"🧠 Retrieving memories for {tenant_id}...")
    elif state == 'ready':
        print(f"✅ Ready to respond to {tenant_id}")

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",  # Default model in litellm format
    memory_model="openai/gpt-4o-mini",  # Memory model in litellm format
    embedding_model="openai/text-embedding-3-small",  # Embedding model in litellm format
    enable_memory=True,  # Enable/disable memory features (default True)
    on_thinking=on_thinking,  # Optional callback for processing states
    debug=False,  # Enable detailed debug logging when needed
    storage_dir="tenants",  # Default base directory for tenant data
    enable_global_memory=False,  # Enable access to _global tenant from all tenants
)

app = FastAPI()
app.include_router(proxy.router, prefix="/v1")
```

Now any OpenAI SDK can point to:

```
http://localhost:8000/v1/<tenant>/chat/completions
```

---

## 🛠️ Configuration Options

The `BrainProxy` class accepts the following parameters:

```python
BrainProxy(
    # Core model settings
    default_model="openai/gpt-4o-mini",  # Primary completion model (litellm format)
    
    # Memory settings
    enable_memory=True,  # Enable/disable memory system
    memory_model="openai/gpt-4o-mini",  # Model for memory management (litellm format)
    embedding_model="openai/text-embedding-3-small",  # Model for embeddings (litellm format)
    mem_top_k=6,  # Maximum number of memories to retrieve per query
    mem_working_max=12,  # Maximum memories to keep in working memory
    enable_global_memory=False,  # Enable access to _global tenant from all tenants
    
    # Storage settings
    storage_dir="tenants",  # Base directory for tenant data
    
    # Customization
    extract_text=None,  # Custom text extraction function for files
    system_prompt=None,  # Optional global system prompt for all conversations
    temporal_awareness=True,  # Enable time-based memory filtering for temporal queries
    
    # Session management (NEW)
    enable_session_memory=True,  # Enable ephemeral session support
    session_ttl_hours=24,  # Session lifetime in hours
    session_max_messages=100,  # Max messages before forced summarization
    session_summarize_after=50,  # Trigger summarization after N messages
    session_memory_max_mb=10.0,  # Max memory usage per session
    on_session_end=None,  # Callback when session expires (tenant_id, session_data)
    
    # Hooks
    manager_fn=None,  # Multi-agent manager hook
    auth_hook=None,  # Authentication hook
    usage_hook=None,  # Usage tracking hook
    on_thinking=None,  # Callback (tenant_id, state) for 'thinking'/'ready' states
    
    # File handling
    max_upload_mb=20,  # Maximum file upload size in MB
    
    # Debugging
    debug=False,  # Enable detailed debug logging
)
```

### 🛠️ Tool Management and Response Quality

#### Tool Filtering System

The `tool_filtering_model` parameter enables smart tool filtering, allowing you to use a large number of tools without degrading model performance:

```python
proxy = BrainProxy(
    default_model="openai/gpt-4o",
    tool_filtering_model="openai/gpt-3.5-turbo",  # Faster model for tool filtering
)
```

Benefits:
- No limit on the number of available tools
- Improved response quality by pre-filtering irrelevant tools
- More efficient model usage by only sending relevant tools
- Better tool selection accuracy

#### Dynamic Temperature

Brain-proxy now automatically adjusts the temperature parameter based on the number of active tools being used. This feature:
- Optimizes response creativity vs precision
- Adapts to the complexity of available tools
- Improves overall response quality
- Requires no manual configuration

This dynamic temperature adjustment helps maintain high-quality responses even when working with multiple tools.

### 🧠 Memory Settings Explained

#### Default Models

BrainProxy uses these default models if not explicitly specified:
- `default_model`: "openai/gpt-4o-mini" - Used for chat completions
- `memory_model`: "openai/gpt-4o-mini" - Used for memory extraction and management
- `embedding_model`: "openai/text-embedding-3-small" - Used for vector embeddings

These are all optional parameters - if you don't specify them, the default values will be used.

#### System Prompt

You can set a global system prompt that will be applied to all conversations:

```python
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    system_prompt="You are Claude, a friendly and helpful AI assistant. You are concise, respectful, and you always maintain a warm, conversational tone. You prefer to explain concepts using analogies and examples."
)
```

This system prompt is applied in a complementary way - it doesn't overwrite system prompts added by the memory or RAG processes. If there's already a system message at the beginning of the conversation, the global system prompt will be prepended to it. Otherwise, a new system message will be added.

#### API Key Requirements

Since brain-proxy uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, you need to set the appropriate API keys as environment variables for your chosen providers:

```bash
# OpenAI models (for openai/gpt-4o, openai/text-embedding-3-small, etc.)
export OPENAI_API_KEY=sk-...

# Anthropic models (for anthropic/claude-3-opus, etc.)
export ANTHROPIC_API_KEY=sk-ant-...

# Azure OpenAI models (for azure/gpt-4, etc.)
export AZURE_API_KEY=...
export AZURE_API_BASE=...
export AZURE_API_VERSION=...

# Google models (for google/gemini-pro, etc.)
export GOOGLE_API_KEY=...
```

You only need to set the API keys for the providers you're actually using. For example, if you're only using OpenAI models, you only need to set `OPENAI_API_KEY`.

See the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for a full list of supported providers and their required environment variables.

### 🧠 Memory Settings Explained

#### `memory_model` - Your Agent's Memory Engine

The `memory_model` parameter specifies which LLM powers your agent's memory capabilities. This model is responsible for:

- Extracting important facts from conversations
- Creating structured memory entries
- Consolidating related memories to avoid duplication

```python
# Using GPT-4o for more advanced memory extraction
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    memory_model="openai/gpt-4o",  # More advanced model for memories
)

# Budget-friendly memory setup
proxy = BrainProxy(
    default_model="openai/gpt-4o", 
    memory_model="openai/gpt-3.5-turbo",  # Economical memory model
)

# Using Anthropic's Claude for memory management
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    memory_model="anthropic/claude-3-haiku-20240307",
)
```

A more capable memory model results in:
- More nuanced memory extraction
- Better recognition of implicit preferences
- Higher quality context preservation

#### `embedding_model` - The Retrieval Brain

This model converts text into vector embeddings for similarity search. It powers:
- Document and memory retrieval
- Similar question matching
- Semantic search across all tenant data

```python
# Using OpenAI's latest embeddings model
proxy = BrainProxy(
    embedding_model="openai/text-embedding-3-large",  # Higher dimension embeddings
)

# Using cost-effective models
proxy = BrainProxy(
    embedding_model="openai/text-embedding-3-small",  # More economical
)

# Azure deployment example
proxy = BrainProxy(
    embedding_model="azure/text-embedding-ada-002", 
)
```

### ⏰ Temporal Awareness

The `temporal_awareness` parameter (default: `True`) enables the agent to understand and respond to time-based queries by intelligently filtering memories based on timestamps.

When enabled, the agent can:
- Understand relative time expressions like "yesterday," "last week," or "next month"
- Filter memories based on when they were created
- Respond accurately to questions about what happened during specific time periods

#### How It Works

When a user asks a question with temporal references like "What did I do yesterday?" or "What are my plans for next month?", the system:

1. Detects the temporal expression in the query
2. Converts it to a specific time range
3. Filters memories that have timestamps within that range
4. Returns only the relevant memories for that time period

This creates a more natural conversational experience as the AI can maintain chronological awareness of events and respond appropriately to time-based queries.

#### Example Implementation

```python
from fastapi import FastAPI
from brain_proxy import BrainProxy
import dotenv

# Load environment variables
dotenv.load_dotenv()

app = FastAPI()

# Initialize BrainProxy with temporal_awareness enabled
brain_proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    memory_model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    enable_memory=True,
    temporal_awareness=True,  # Enable time-based memory filtering
    debug=True,  # Set to True to see detailed logs
)

app.include_router(brain_proxy.router, prefix="/v1")

@app.get("/")
def root():
    return {
        "message": "Brain-proxy with temporal awareness is running!",
        "models": {
            "default": brain_proxy.default_model,
            "memory": brain_proxy.memory_model,
            "embedding": brain_proxy.embedding_model
        },
        "temporal_awareness": brain_proxy.temporal_awareness
    }
```

#### Testing Temporal Awareness

Here's how you can test the temporal awareness feature with curl commands:

```bash
# 1. Store a memory about something in the past
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Remember that I bought a car last week."}
  ]
}'

# 2. Store a memory about something happening today
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Today I started learning Python programming."}
  ]
}'

# 3. Store a memory about future plans
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "I plan to visit Japan next month for a vacation."}
  ]
}'

# 4. Query about past events
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What happened last week?"}
  ]
}'
# Response will mention the car purchase

# 5. Query about today's activities
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What did I do today?"}
  ]
}'
# Response will mention Python programming

# 6. Query about future plans
curl -X POST http://localhost:8000/v1/my_tenant/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What are my plans for next month?"}
  ]
}'
# Response will mention the Japan vacation
```

This feature significantly enhances the contextual awareness of conversations by providing chronologically accurate responses to time-based queries.

### 📄 Custom Text Extraction

The `extract_text` parameter lets you plug in specialized text extraction functions for different file types.

#### PDF Extraction Example

```python
from pdfminer.high_level import extract_text

def extract_document_text(path, mime_type):
    """Extract text from various document formats"""
    if mime_type == "application/pdf":
        return extract_text(path)
    elif mime_type == "text/plain":
        return path.read_text(encoding="utf-8")
    elif mime_type == "text/csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_string()
    else:
        return f"Unsupported format: {mime_type}"

# Use the custom extractor
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    extract_text=extract_document_text
)
```

#### Advanced Image + Document Extraction

```python
async def multimodal_extractor(path, mime_type):
    """Extract text from documents and images using specialized models"""
    if mime_type.startswith("image/"):
        # Use Moondream (open-source lightweight vision model) for images
        try:
            import moondream as md
            from PIL import Image
            
            # Load the image
            img = Image.open(path)
            
            # Initialize Moondream model
            # You can use either the 2B parameter model or the smaller 0.5B model
            model = md.vl(model="path/to/moondream-2b-int8.mf")
            
            # Encode the image (this is a crucial step for Moondream)
            encoded_image = model.encode_image(img)
            
            # Generate a descriptive caption
            caption = model.caption(encoded_image)["caption"]
            
            # You can also ask specific questions about the image
            # details = model.query(encoded_image, "Describe this image in detail.")["answer"]
            
            return f"Image description: {caption}"
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    elif mime_type == "application/pdf":
        # Extract text from PDFs
        from pdfminer.high_level import extract_text
        return extract_text(path)
    
    # Handle other formats...

This example uses [Moondream](https://github.com/vikhyat/moondream), an efficient multimodal vision model that can be run entirely locally, even on CPU-only machines. Install it with `pip install moondream`.

### 🪝 Powerful Hook Functions

brain-proxy provides three powerful hooks that help you customize, secure, and monitor your proxy:

#### `auth_hook` - Custom Authentication

Secure your endpoints with tenant-specific authentication:

```python
async def custom_auth(request, tenant):
    """Validate tenant-specific access"""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    # Check tenant-specific permissions
    if not is_authorized(token, tenant):
        raise HTTPException(status_code=403, detail="Not authorized for this tenant")
    
    # You can also map tokens to specific users
    request.state.user_id = get_user_id(token)

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    auth_hook=custom_auth
)
```

#### `usage_hook` - Track LLM Consumption

Monitor token usage and costs by tenant:

```python
async def track_usage(tenant, tokens, duration):
    """Record usage metrics per tenant"""
    print(f"Tenant {tenant} used {tokens} tokens in {duration:.2f}s")
    
    # Log to database
    await db.usage_logs.insert_one({
        "tenant": tenant,
        "tokens": tokens,
        "duration": duration,
        "timestamp": datetime.now(),
        "cost": calculate_cost(tokens)
    })
    
    # Update quota limits
    await update_tenant_quota(tenant, tokens)

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    usage_hook=track_usage
)
```

#### `manager_fn` - Multi-Agent Orchestration

This hook allows you to add multi-agent workflows for coordinating different AI models or systems:

```python
async def manager_fn(request, tenant, conversation):
    """Custom processing logic for each tenant/request"""
    # You can inspect the request and tenant to determine special handling
    
    # Perform custom agent routing or orchestration
    if "financial" in request.body:
        # Route to specialized financial analysis
        return await financial_agent.process(conversation)
    
    # You can return processed messages or modify the conversation flow
    # The output of this function is used in the processing pipeline
    
    # Return None for default behavior
    return None

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    manager_fn=manager_fn
)
```

The `manager_fn` hook is primarily designed for integrating with more complex agent frameworks or enabling custom message preprocessing before the chat completion is generated.

With these hooks, you can build sophisticated multi-tenant applications with fine-grained security, usage monitoring, and dynamic agent delegation.

### 🎯 `on_thinking` Callback - Real-time Processing Feedback

The `on_thinking` callback provides real-time feedback about the processing state, perfect for updating UI loading states and animations.

#### Callback Signature

```python
on_thinking: Optional[Callable[[str, str], Any]] = None
```

- **tenant_id** (str): The tenant identifier for the current request
- **state** (str): The current processing state
  - `'thinking'`: Triggered before memory retrieval starts (only when memory is enabled)
  - `'ready'`: Triggered just before the response is sent (both streaming and non-streaming)

#### State Flow

```
Request arrives
    ↓
[thinking] → Memory retrieval starts
    ↓
Processing with LLM
    ↓
[ready] → Response begins (streaming or complete)
```

#### Example: UI Loading Animation

```python
async def handle_thinking_state(tenant_id: str, state: str):
    """Control UI loading animations based on processing state"""
    
    if state == 'thinking':
        # Start loading animation
        await send_websocket_message(tenant_id, {
            "type": "status",
            "state": "thinking",
            "message": "Retrieving context and memories..."
        })
        # Show spinner, skeleton loader, etc.
        
    elif state == 'ready':
        # Stop loading animation, prepare for content
        await send_websocket_message(tenant_id, {
            "type": "status", 
            "state": "ready",
            "message": "Processing complete"
        })
        # Hide spinner, prepare content area

proxy = BrainProxy(
    on_thinking=handle_thinking_state,
    enable_memory=True
)
```

#### Example: Performance Monitoring

```python
class PerformanceTracker:
    def __init__(self):
        self.timings = {}
    
    def track_state(self, tenant_id: str, state: str):
        """Track processing time between states"""
        import time
        
        if state == 'thinking':
            self.timings[tenant_id] = time.time()
            print(f"⏱️ [{tenant_id}] Memory retrieval started")
            
        elif state == 'ready':
            if tenant_id in self.timings:
                duration = time.time() - self.timings[tenant_id]
                print(f"✅ [{tenant_id}] Ready in {duration:.2f}s")
                # Log to monitoring service, metrics dashboard, etc.

tracker = PerformanceTracker()
proxy = BrainProxy(
    on_thinking=tracker.track_state,
    enable_memory=True
)
```

#### Synchronous vs Asynchronous

The callback supports both synchronous and asynchronous functions:

```python
# Synchronous callback
def sync_callback(tenant_id: str, state: str):
    if state == 'thinking':
        logger.info(f"Thinking for {tenant_id}")
    elif state == 'ready':
        logger.info(f"Ready for {tenant_id}")

# Asynchronous callback  
async def async_callback(tenant_id: str, state: str):
    if state == 'thinking':
        await async_operation(tenant_id, "thinking")
    elif state == 'ready':
        await async_operation(tenant_id, "ready")

# Both work seamlessly
proxy = BrainProxy(on_thinking=sync_callback)  # or async_callback
```

#### Key Benefits

- **Improved UX**: Users see immediate feedback that their request is being processed
- **Streaming Support**: Works identically for both streaming and non-streaming responses
- **Error Resilient**: Callback errors are caught and logged without breaking the main flow
- **Lightweight**: Minimal overhead, called only twice per request
- **Flexible**: Use for animations, monitoring, logging, or any custom state management

---

## 🧠 Multi-tenancy explained

Every tenant (`/v1/acme`, `/v1/alpha`, etc):

- Gets its own vector store (for RAG)
- Has isolated LangMem memory (short- and long-term)
- Can upload files (auto-indexed + persisted)
- Has a dedicated file storage directory structure

This means you can serve multiple brands or users safely and scalably from a single backend.

---

## 💬 LiteLLM/"OpenAI" SDK Example

```python
import openai

openai.api_key = "sk-fake"
openai.base_url = "http://localhost:8000/v1/acme"

response = openai.ChatCompletion.create(
    model="openai/gpt-4o",  # Now specify provider/model!
    messages=[{"role": "user", "content": "What's 3 + 2?"}]
)

print(response["choices"][0]["message"]["content"])
```

### Function Calling

brain-proxy supports OpenAI-compatible function calling through the tools parameter in requests:

```python
response = openai.ChatCompletion.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "What time is it in UTC?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in UTC",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }]
)

# The model may respond with a function call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function called: {tool_call.function.name}")
```

### Streaming:

```python
stream = openai.ChatCompletion.create(
    model="openai/gpt-4o",  # Or e.g. "cerebras/llama3-70b-instruct"
    stream=True,
    messages=[{"role": "user", "content": "Tell me a short story about an AI fox."}]
)

for chunk in stream:
    print(chunk.choices[0].delta.get("content", ""), end="")
```

---

## 🔗 LangChain Integration

brain-proxy now provides a LangChain-compatible model interface, making it easy to use in LangChain-based applications and frameworks like CrewAI and LangGraph.

### Basic Usage

The most common way to use brain-proxy with LangChain is to connect to an existing brain-proxy service:

```python
from brain_proxy import BrainProxyLangChainModel
from langchain.chains import ConversationChain

# Create LangChain model pointing to brain-proxy service
model = BrainProxyLangChainModel(
    tenant="my_tenant",
    base_url="http://localhost:8000/v1",  # Optional, this is the default
    model="anthropic/claude-3-opus",  # Optional, uses brain-proxy default if not specified
    streaming=True  # Optional
)

# Use in any LangChain application
chain = ConversationChain(llm=model)
response = await chain.ainvoke({"input": "Hello, how are you?"})
```

### Advanced Agent Usage

Here's how to use brain-proxy with LangChain's agent framework:

```python
from brain_proxy import BrainProxyLangChainModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool

# Create model instance
model = BrainProxyLangChainModel(
    tenant="my_tenant",
    base_url="https://your-brain-proxy.com/v1",  # Point to your brain-proxy service
    streaming=True
)

# Configure your agent and tools
tools = [...]  # Your tools here
agent = create_openai_tools_agent(model, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
result = await agent_executor.ainvoke({"input": "What's the weather like?"})
```

### Direct Instance Usage (Advanced)

For advanced use cases, you can also create a BrainProxyLangChainModel from a local BrainProxy instance:

```python
from brain_proxy import BrainProxy, BrainProxyLangChainModel

# Initialize BrainProxy
brain_proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    system_prompt="You are a helpful AI assistant"
)

# Create LangChain model from instance
model = BrainProxyLangChainModel(
    tenant="my_tenant",
    brain_proxy=brain_proxy,
    streaming=True
)
```

The LangChain integration supports:
- Streaming responses with proper callback handling
- Memory and RAG features through brain-proxy's built-in capabilities
- Multi-tenant isolation
- All LiteLLM-supported models
- Async-first design for optimal performance

---

## ⚡️ Model selection

By default, brain-proxy now uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood. This means you can specify any supported model using the `provider/model` format:

- `openai/gpt-4o`
- `cerebras/llama3-70b-instruct`
- `anthropic/claude-3-opus-20240229`
- ...and many more!

Just set the `model` parameter in your requests accordingly.

---

## 📎 File Uploads

Send `file_data` parts inside messages to upload PDFs, CSVs, images, etc:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Here's a report:" },
    { "type": "file_data", "file_data": {
        "name": "report.pdf",
        "mime": "application/pdf",
        "data": "...base64..."
    }}
  ]
}
```

Files are saved in tenant-specific directories, parsed, embedded, and used in RAG on the fly.

---

## 🛠️ Tools Support

brain-proxy now includes a powerful tool system that makes it easy to add custom functionality to your AI assistant. Tools can be defined using a simple decorator:

```python
from brain_proxy import tool

@tool(description="Get the current weather for a location")
async def get_weather(location: str) -> dict:
    """Get current weather conditions.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    
    Returns:
        dict: Weather information including temperature and conditions
    """
    return {
        "temperature": "72°F",
        "condition": "sunny"
    }

# Tools are automatically registered with BrainProxy
proxy = BrainProxy()
```

The tool system features:
- Automatic parameter schema generation from type hints and docstrings
- Support for both sync and async functions
- Global tool registry for easy reuse
- Compatible with OpenAI function calling format

You can also disable automatic tool registration if needed:
```python
proxy = BrainProxy(use_registry_tools=False)
```

## � Streaming & Multi-Tool Support

brain-proxy now features robust support for streaming responses with multiple tool calls, making it perfect for complex, interactive AI applications. The streaming system has been completely redesigned to handle:

- Multiple concurrent tool calls within a single streaming response
- Index-based tracking for reliable tool call ordering
- Robust argument accumulation across stream chunks
- Proper preservation of tool call IDs
- Enhanced tool call structure with smart defaults

Example of handling streamed multi-tool responses:

```python
from brain_proxy import BrainProxy, tool

@tool(description="Search the web")
async def search_web(query: str) -> str:
    return f"Results for: {query}"

@tool(description="Analyze sentiment")
async def analyze_sentiment(text: str) -> str:
    return "positive"

proxy = BrainProxy()

# The LLM can now make multiple tool calls in a single streaming response
# Each tool call is properly tracked and managed, even when split across chunks
response = await proxy.chat.completions.create(
    messages=[{"role": "user", "content": "Search for latest news and analyze their sentiment"}],
    stream=True
)

async for chunk in response:
    # Tool calls are automatically tracked and managed
    print(chunk)
```

The improved streaming system ensures reliable handling of complex interactions where the AI needs to:
1. Make multiple tool calls in sequence
2. Process tool results while streaming
3. Maintain context across stream chunks
4. Handle parallel tool executions

This makes brain-proxy ideal for building sophisticated AI applications that require real-time interaction and complex tool usage.

## �📑 Custom Document Processing

The `extract_text` function now supports returning either a string or a list of LangChain `Document` objects:

```python
from langchain.schema import Document

def process_document(path: Path, mime_type: str) -> str | List[Document]:
    """Custom document processor that can return string or Documents"""
    if mime_type == "application/pdf":
        # Return a list of Documents with metadata
        return [
            Document(
                page_content="Page 1 content...",
                metadata={"page": 1, "source": path.name}
            ),
            Document(
                page_content="Page 2 content...",
                metadata={"page": 2, "source": path.name}
            )
        ]
    else:
        # Return simple string for other formats
        return "Extracted text content..."

proxy = BrainProxy(extract_text=process_document)
```

## 🌐 Global Memory

Enable shared memory across all tenants with the `enable_global_memory` flag:

```python
proxy = BrainProxy(
    enable_global_memory=True  # Allows all tenants to access _global memories
)
```

When enabled:
- Any tenant can read from the `_global` tenant's memory
- Useful for shared knowledge bases or company-wide information
- Individual tenant memories remain private

## ⚡ Vector Store Improvements

The Upstash vector store adapter now uses LangChain's native Upstash integration for better performance and reliability:

```python
proxy = BrainProxy(
    # Upstash configuration (uses LangChain integration)
    upstash_rest_url="https://your-instance.upstash.io",
    upstash_rest_token="your-token"
)
```

Benefits:
- Improved query performance
- Better connection handling
- Native LangChain compatibility
- Simplified configuration

---

## 🧾 Custom PDF extractor example

```python
from pdfminer.high_level import extract_text

def parse_pdf(path: Path, mime: str) -> str:
    """Custom PDF extractor"""
    if mime == "application/pdf":
        return extract_text(path)
    return "(unsupported format)"

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    extract_text=parse_pdf
)
```

---

## 🐞 Debugging

Enable debug mode to see detailed information about memory processing, file ingestion, and other operations:

```python
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    debug=True  # Shows detailed logs for troubleshooting
)
```

---

## 📦 Roadmap

- [x] Multi-agent manager hook
- [x] Usage hooks + token metering
- [x] Use LiteLLM instead to support more models
- [x] Tenant-specific file storage
- [x] Debug mode for troubleshooting
- [ ] MCP support
- [ ] LangGraph integration

---

## ⚖️ License

MIT — free to use, fork, and build on.  
Made for backend devs who want to move fast ⚡

---

## ❤️ Contributing

Issues and PRs welcome!

Let's build smarter backends — together.
