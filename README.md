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
- ✅ No frontend changes required
- ✅ **Now uses LiteLLM by default — specify any model using `provider/model` (e.g., `openai/gpt-4o`, `cerebras/llama3-70b-instruct`)**

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

proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",  # Default model in litellm format
    memory_model="openai/gpt-4o-mini",  # Memory model in litellm format
    embedding_model="openai/text-embedding-3-small",  # Embedding model in litellm format
    enable_memory=True,  # Enable/disable memory features (default True)
    debug=False,  # Enable detailed debug logging when needed
    storage_dir="tenants",  # Default base directory for tenant data
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
    
    # Storage settings
    storage_dir="tenants",  # Base directory for tenant data
    
    # Customization
    extract_text=None,  # Custom text extraction function for files
    system_prompt=None,  # Optional global system prompt for all conversations
    temporal_awareness=True,  # Enable time-based memory filtering for temporal queries
    
    # Hooks
    manager_fn=None,  # Multi-agent manager hook
    auth_hook=None,  # Authentication hook
    usage_hook=None,  # Usage tracking hook
    
    # File handling
    max_upload_mb=20,  # Maximum file upload size in MB
    
    # Debugging
    debug=False,  # Enable detailed debug logging
)
```

### ⚠️ Important: API Keys and Default Models

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

## 🧾 Custom PDF extractor example

```python
from pdfminer.high_level import extract_text

def parse_pdf(path: Path, mime: str) -> str:
    if mime == "application/pdf":
        return extract_text(path)
    return "(unsupported format)"
```

```python
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
