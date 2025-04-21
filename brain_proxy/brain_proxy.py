"""
brain_proxy.py  —  FastAPI / ASGI router with LangMem + Chroma

pip install fastapi openai langchain-chroma langmem tiktoken
"""

from __future__ import annotations
import asyncio, base64, hashlib, json, time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
#from langchain_litellm import ChatLiteLLM
from litellm import acompletion
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# LangMem primitives (functions, not classes)
from langmem import create_memory_manager

# -------------------------------------------------------------------
# Pydantic schemas (OpenAI spec + file‑data part)
# -------------------------------------------------------------------
class FileData(BaseModel):
    name: str
    mime: str
    data: str  # base‑64 bytes


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    file_data: Optional[FileData] = Field(None, alias="file_data")


class ChatMessage(BaseModel):
    role: str
    content: str | List[ContentPart]


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def _maybe(fn, *a, **k):
    return await fn(*a, **k) if asyncio.iscoroutinefunction(fn) else fn(*a, **k)


# -------------------------------------------------------------------
# BrainProxy
# -------------------------------------------------------------------
def chroma_vec_factory(collection_name: str, embeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=f".chroma/{collection_name}",
        embedding_function=embeddings,
    )

def default_vector_store_factory(tenant, embeddings):
    return chroma_vec_factory(f"vec_{tenant}", embeddings)

class BrainProxy:
    """Drop‑in OpenAI‑compatible proxy with Chroma + LangMem memory"""

    def __init__(
        self,
        *,
        openai_api_key: str,
        vector_store_factory: Callable[[str, Any], Chroma] = default_vector_store_factory,
        # memory settings
        enable_memory: bool = True,
        memory_model: str = "gpt-3.5-turbo",
        mem_top_k: int = 6,
        mem_working_max: int = 12,
        # misc
        upstream_base: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o",
        storage_dir: str | Path = "tenants",
        extract_text: Callable[[Path, str], str] | None = None,
        manager_fn: Callable[..., Any] | None = None,  # multi‑agent hook
        auth_hook: Callable[[Request, str], Any] | None = None,
        usage_hook: Callable[[str, int, float], Any] | None = None,
        max_upload_mb: int = 20,
    ):
        #self.llm = AsyncOpenAI(api_key=openai_api_key, base_url=upstream_base)
        #self.base_url = upstream_base
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.vec_factory = lambda tenant: vector_store_factory(tenant, self.embeddings)
        self.enable_memory = enable_memory
        self.memory_model = memory_model
        self.mem_top_k = mem_top_k
        self.mem_working_max = mem_working_max

        self.default_model = default_model
        self.storage_dir = Path(storage_dir)
        self.extract_text = extract_text or (
            lambda p, m: p.read_text("utf-8", "ignore")
        )
        self.manager_fn = manager_fn
        self.auth_hook = auth_hook
        self.usage_hook = usage_hook
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

        self._mem_managers: Dict[str, Any] = {}  # cached per tenant
        self._working_memory: Dict[str, List[str]] = {}

        self.router = APIRouter()
        self._mount()

    # ----------------------------------------------------------------
    # Memory helpers
    # ----------------------------------------------------------------
    def _get_mem_manager(self, tenant: str):
        if not self.enable_memory:
            return None
        if tenant in self._mem_managers:
            return self._mem_managers[tenant]

        # use the tenant’s chroma collection for memory as well
        vec = self.vec_factory(f"{tenant}_memory")
        async def _search_mem(query: str, k: int):
            docs = vec.similarity_search(query, k=k)
            return [d.page_content for d in docs]

        async def _store_mem(memories: List[Any]):
            # Accepts a list of Memory objects or strings, store as documents
            docs = [Document(page_content=m.content if hasattr(m, 'content') else m) for m in memories]
            vec.add_documents(docs)

        # manager is an async callable
        manager = create_memory_manager(self.memory_model)

        self._mem_managers[tenant] = (manager, _search_mem, _store_mem)
        return self._mem_managers[tenant]

    async def _retrieve_memories(self, tenant: str, user_text: str) -> str:
        manager_tuple = self._get_mem_manager(tenant)
        if not manager_tuple:
            return ""
        _, search, _ = manager_tuple
        memories = await search(user_text, k=self.mem_top_k)
        return "\n".join(memories)

    async def _write_memories(
        self, tenant: str, conversation: List[Dict[str, Any]]
    ):
        manager_tuple = self._get_mem_manager(tenant)
        if not manager_tuple:
            return
        manager, _, store = manager_tuple
        # call manager with current conversation to extract memories
        extracted = await manager(conversation)
        if extracted:
            memories = [m[1] for m in extracted]  # second item is the text
            await store(memories)

    # ----------------------------------------------------------------
    # File helpers
    # ----------------------------------------------------------------
    def _split_files(
        self, msgs: List[ChatMessage]
    ) -> tuple[List[Dict[str, Any]], List[FileData]]:
        files, cleaned = [], []
        for m in msgs:
            if isinstance(m.content, str):
                cleaned.append(m.model_dump())
                continue
            parts = []
            for p in m.content:
                if p.type == "file_data" and p.file_data:
                    files.append(p.file_data)
                    parts.append(
                        ContentPart(
                            type="text",
                            text=f"[file {p.file_data.name} ingested]",
                        ).model_dump()
                    )
                else:
                    parts.append(p.model_dump())
            cleaned.append({"role": m.role, "content": parts})
        return cleaned, files

    async def _ingest_files(self, files: List[FileData], tenant: str):
        up_dir = self.storage_dir / tenant / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        vec = self.vec_factory(tenant)

        new_docs = []
        for f in files:
            blob = base64.b64decode(f.data)
            if len(blob) > self.max_upload_bytes:
                raise HTTPException(413, f"{f.name} exceeds upload limit")
            digest = _sha(blob)
            path = up_dir / f"{digest}-{f.name}"
            if path.exists():
                continue
            path.write_bytes(blob)
            text = await _maybe(self.extract_text, path, f.mime)
            new_docs.append(Document(page_content=text))
        if new_docs:
            await _maybe(vec.add_documents, new_docs)

    # ----------------------------------------------------------------
    # RAG helper
    # ----------------------------------------------------------------
    async def _rag(self, msgs: List[Dict[str, Any]], tenant: str, k: int = 4):
        vec = self.vec_factory(tenant)
        query = (
            msgs[-1]["content"]
            if isinstance(msgs[-1]["content"], str)
            else next(p["text"] for p in msgs[-1]["content"] if p["type"] == "text")
        )
        docs = await _maybe(vec.similarity_search, query, k)
        if not docs:
            return msgs
        ctx = "\n\n".join(d.page_content for d in docs)
        return msgs[:-1] + [
            {"role": "system", "content": f"Context documents:\n{ctx}"},
            msgs[-1],
        ]

    # ----------------------------------------------------------------
    # Upstream dispatch
    # ----------------------------------------------------------------
    async def _dispatch_bak(self, msgs, model: str, *, stream: bool):
        """Dispatch to OpenAI API"""
        if stream:
            return await self.llm.chat.completions.create(
                model=model, messages=msgs, stream=stream
            )
        else:
            # For non-streaming responses, we need to await the response directly
            return await self.llm.chat.completions.create(
                model=model, messages=msgs, stream=False
            )

    async def _dispatch(self, msgs, model: str, *, stream: bool):
        """Dispatch to OpenAI API"""
        if stream:
            return await acompletion(
                model=model, messages=msgs, stream=stream,
                #base_url=self.base_url
            )
        else:
            # For non-streaming responses, we need to await the response directly
            return await acompletion(
                model=model, messages=msgs, stream=False,
                #base_url=self.base_url
            )


    # ----------------------------------------------------------------
    # FastAPI route
    # ----------------------------------------------------------------
    def _mount(self):
        @self.router.post("/{tenant}/chat/completions")
        async def chat(request: Request, tenant: str):
            if self.auth_hook:
                await _maybe(self.auth_hook, request, tenant)

            body = await request.json()
            req = ChatRequest(**body)
            msgs, files = self._split_files(req.messages)

            if files:
                await self._ingest_files(files, tenant)

            # LangMem retrieve
            if self.enable_memory:
                user_text = (
                    msgs[-1]["content"]
                    if isinstance(msgs[-1]["content"], str)
                    else next(
                        p["text"] for p in msgs[-1]["content"] if p["type"] == "text"
                    )
                )
                mem_block = await self._retrieve_memories(tenant, user_text)
                if mem_block:
                    msgs = msgs[:-1] + [
                        {
                            "role": "system",
                            "content": "Relevant memories:\n" + mem_block,
                        },
                        msgs[-1],
                    ]

            msgs = await self._rag(msgs, tenant)

            upstream_iter = await self._dispatch(
                msgs, req.model or self.default_model, stream=req.stream
            )
            t0 = time.time()

            if not req.stream:
                # No need to await here since _dispatch already returns the complete response
                response_data = upstream_iter.model_dump()
                await self._write_memories(tenant, msgs + [upstream_iter.choices[0].message.model_dump()])
                if self.usage_hook and upstream_iter.usage:
                    await _maybe(
                        self.usage_hook,
                        tenant,
                        upstream_iter.usage.total_tokens,
                        time.time() - t0,
                    )
                return JSONResponse(response_data)

            # streaming path
            async def event_stream() -> AsyncIterator[str]:
                tokens = 0
                buf: List[str] = []
                async for chunk in upstream_iter:
                    payload = json.loads(chunk.model_dump_json())
                    delta = payload["choices"][0].get("delta", {}).get("content", "")
                    if delta is None:
                        delta = ""
                    tokens += len(delta)
                    buf.append(delta)
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                await self._write_memories(
                    tenant, msgs + [{"role": "assistant", "content": "".join(buf)}]
                )
                if self.usage_hook:
                    await _maybe(self.usage_hook, tenant, tokens, time.time() - t0)

            return StreamingResponse(event_stream(), media_type="text/event-stream")


# -------------------------------------------------------------------
# Example Chroma factories
# -------------------------------------------------------------------
"""
# Usage
from fastapi import FastAPI
from brain_proxy import BrainProxy

proxy = BrainProxy(
    openai_api_key="sk-…",
)

app = FastAPI()
app.include_router(proxy.router, prefix="/v1")

# Point any OpenAI SDK at
# http://localhost:8000/v1/<tenant>/chat/completions
# Upload files via messages[].content[].file_data
# Enjoy RAG + LangMem without extra DBs or infra
"""