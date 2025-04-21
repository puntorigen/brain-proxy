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
    openai_api_key="sk-...",  # used for both LLM + embeddings
)

app = FastAPI()
app.include_router(proxy.router, prefix="/v1")
```

Now any OpenAI SDK can point to:

```
http://localhost:8000/v1/<tenant>/chat/completions
```

---

## 🧠 Multi-tenancy explained

Every tenant (`/v1/acme`, `/v1/alpha`, etc):

- Gets its own vector store (for RAG)
- Has isolated LangMem memory (short- and long-term)
- Can upload files (auto-indexed + persisted)

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

Files are saved, parsed, embedded, and used in RAG on the fly.

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
    openai_api_key="sk-...",
    extract_text=parse_pdf
)
```

---

## 📦 Roadmap

- [x] Multi-agent manager hook
- [x] Usage hooks + token metering
- [x] Use LiteLLM instead to support more models
- [ ] MCP support
- [ ] LangGraph integration

---

## ⚖️ License

MIT — free to use, fork, and build on.  
Made for backend devs who want to move fast ⚡

---

## ❤️ Contributing

Issues and PRs welcome!

Let’s build smarter backends — together.
