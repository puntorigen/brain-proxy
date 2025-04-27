"""
brain_proxy_langchain.py — LangChain integration for BrainProxy

This module provides a LangChain-compatible model interface for BrainProxy,
allowing it to be used in LangChain-based applications and frameworks.
"""

from typing import Any, Dict, List, Optional, Iterator, Union
from datetime import datetime, timezone
import json
import httpx

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from .brain_proxy import BrainProxy, ChatMessage, ChatRequest


class BrainProxyLangChainModel(BaseChatModel):
    """LangChain interface for BrainProxy chat model.
    
    This class allows using BrainProxy as a LangChain chat model, enabling
    integration with LangChain-based applications and frameworks like CrewAI
    and LangGraph.

    Args:
        tenant: The tenant name to use for chat completions.
        brain_proxy: Optional BrainProxy instance. If not provided, will make HTTP calls to base_url.
        base_url: Base URL for BrainProxy API. Defaults to "http://localhost:8000/v1".
        model: Optional model name to use. If not provided, uses BrainProxy's default.
        streaming: Whether to stream the responses.
    """

    tenant: str
    """The tenant name to use for chat completions."""

    brain_proxy: Optional[BrainProxy] = None
    """Optional BrainProxy instance for direct calls."""

    base_url: str = "http://localhost:8000/v1"
    """Base URL for BrainProxy API when not using direct instance."""

    model: Optional[str] = None
    """Optional model name to use. If not provided, uses BrainProxy's default."""

    streaming: bool = False
    """Whether to stream the responses."""

    _http_client: Optional[httpx.AsyncClient] = None
    """Lazy-loaded HTTP client for API calls."""

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    def _convert_to_chat_messages(self, messages: List[BaseMessage]) -> List[ChatMessage]:
        """Convert LangChain messages to BrainProxy ChatMessage format."""
        converted = []
        for msg in messages:
            role = {
                SystemMessage: "system",
                HumanMessage: "user",
                AIMessage: "assistant"
            }.get(type(msg), "user")
            
            converted.append(ChatMessage(
                role=role,
                content=msg.content,
                timestamp=datetime.now(timezone.utc)
            ))
        return converted

    def _create_chat_result(self, response_content: str) -> ChatResult:
        """Create a ChatResult from a response string."""
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=response_content),
                    text=response_content
                )
            ]
        )

    async def _make_api_request(
        self,
        messages: List[ChatMessage],
        stream: bool = False
    ) -> Union[Dict[str, Any], httpx.Response]:
        """Make HTTP request to BrainProxy API."""
        client = await self._ensure_http_client()
        url = f"{self.base_url}/{self.tenant}/chat/completions"
        
        request_data = {
            "model": self.model,
            "messages": [msg.model_dump() for msg in messages],
            "stream": stream
        }

        if stream:
            response = await client.post(url, json=request_data, stream=True)
            response.raise_for_status()
            return response
        else:
            response = await client.post(url, json=request_data)
            response.raise_for_status()
            return response.json()

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion."""
        chat_messages = self._convert_to_chat_messages(messages)
        
        if self.brain_proxy:
            # Use direct BrainProxy instance
            request = ChatRequest(
                model=self.model,
                messages=chat_messages,
                stream=self.streaming
            )
            response = await self.brain_proxy.chat(request, self.tenant)
            
            if self.streaming:
                # Handle streaming response
                full_response = ""
                async for chunk in response.body_iterator:
                    if chunk.startswith(b"data: "):
                        chunk_data = chunk.decode("utf-8")[6:]  # Remove "data: " prefix
                        if chunk_data.strip() == "[DONE]":
                            break
                        
                        chunk_json = json.loads(chunk_data)
                        content = chunk_json["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            full_response += content
                            if run_manager:
                                await run_manager.on_llm_new_token(content)
                
                return self._create_chat_result(full_response)
            else:
                # Handle non-streaming response
                response_data = response.body
                content = response_data["choices"][0]["message"]["content"]
                return self._create_chat_result(content)
        else:
            # Use HTTP API
            if self.streaming:
                # Handle streaming response
                full_response = ""
                response = await self._make_api_request(chat_messages, stream=True)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]  # Remove "data: " prefix
                        if chunk_data.strip() == "[DONE]":
                            break
                        
                        chunk_json = json.loads(chunk_data)
                        content = chunk_json["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            full_response += content
                            if run_manager:
                                await run_manager.on_llm_new_token(content)
                
                return self._create_chat_result(full_response)
            else:
                # Handle non-streaming response
                response_data = await self._make_api_request(chat_messages)
                content = response_data["choices"][0]["message"]["content"]
                return self._create_chat_result(content)

    async def _agenerate_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        """Async generate streaming chat completion."""
        chat_messages = self._convert_to_chat_messages(messages)
        
        if self.brain_proxy:
            # Use direct BrainProxy instance
            request = ChatRequest(
                model=self.model,
                messages=chat_messages,
                stream=True
            )
            response = await self.brain_proxy.chat(request, self.tenant)
            
            async for chunk in response.body_iterator:
                if chunk.startswith(b"data: "):
                    chunk_data = chunk.decode("utf-8")[6:]  # Remove "data: " prefix
                    if chunk_data.strip() == "[DONE]":
                        break
                    
                    chunk_json = json.loads(chunk_data)
                    content = chunk_json["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        if run_manager:
                            await run_manager.on_llm_new_token(content)
                        yield self._create_chat_result(content)
        else:
            # Use HTTP API
            response = await self._make_api_request(chat_messages, stream=True)
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data.strip() == "[DONE]":
                        break
                    
                    chunk_json = json.loads(chunk_data)
                    content = chunk_json["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        if run_manager:
                            await run_manager.on_llm_new_token(content)
                        yield self._create_chat_result(content)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "brain_proxy"

    async def aclose(self):
        """Close any resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
