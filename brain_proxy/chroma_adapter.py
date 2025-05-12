"""
ChromaDB adapter for brain-proxy.
"""

from typing import List
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ChromaAsyncWrapper:
    """Async wrapper for ChromaDB to maintain consistency with Upstash adapter."""
    
    def __init__(self, collection_name: str, embeddings: Embeddings, max_workers: int = 10):
        """Initialize ChromaDB wrapper.
        
        Args:
            collection_name: Name of the collection
            embeddings: LangChain embeddings interface
            max_workers: Maximum number of threads in the thread pool (default: 10)
        """
        self.chroma = Chroma(
            collection_name=collection_name,
            persist_directory=f".chroma/{collection_name}",
            embedding_function=embeddings,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB asynchronously."""
        await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.chroma.add_documents,
            documents
        )
    
    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Run similarity search asynchronously."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.chroma.similarity_search,
            query,
            k
        )


def chroma_vec_factory(collection_name: str, embeddings: Embeddings, max_workers: int = 10) -> ChromaAsyncWrapper:
    """Create a new async ChromaDB wrapper instance.
    
    Args:
        collection_name: Name of the collection
        embeddings: LangChain embeddings interface
        max_workers: Maximum number of threads in the thread pool (default: 10)
    """
    return ChromaAsyncWrapper(
        collection_name=collection_name,
        embeddings=embeddings,
        max_workers=max_workers
    )
