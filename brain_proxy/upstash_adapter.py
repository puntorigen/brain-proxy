from typing import List, Optional
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from upstash_vector import Index
from datetime import datetime, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class UpstashVectorStore:
    """Upstash vector store adapter for brain-proxy."""

    def __init__(
        self,
        collection_name: str,
        embedding_function: Embeddings,
        rest_url: str,
        rest_token: str,
        max_workers: int = 10,
    ):
        """Initialize Upstash vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_function: LangChain embeddings interface
            rest_url: Upstash REST URL 
            rest_token: Upstash REST token
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Ensure URL has protocol
        if not rest_url.startswith(('http://', 'https://')):
            rest_url = f'https://{rest_url}'
            
        self.index = Index(url=rest_url, token=rest_token)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
            
        # Embed documents
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_function.aembed_documents(texts)

        # Prepare vectors for Upstash
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            vector = {
                "id": f"{self.collection_name}_{datetime.now(timezone.utc).timestamp()}",
                "vector": embedding,
                "metadata": {
                    **doc.metadata,
                    "content": doc.page_content
                }
            }
            vectors.append(vector)

        # Upload to Upstash in batches of 100
        tasks = []
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            # Run synchronous upsert in thread pool
            future = asyncio.get_running_loop().run_in_executor(
                self._executor,
                partial(
                    self.index.upsert,
                    vectors=batch,
                    namespace=self.collection_name
                )
            )
            tasks.append(future)
        
        # Wait for all batches to complete
        await asyncio.gather(*tasks)

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        # Get query embedding
        query_embedding = await self.embedding_function.aembed_query(query)

        # Search Upstash (run sync query in thread pool)
        try:
            results = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                partial(
                    self.index.query,
                    vector=query_embedding,
                    namespace=self.collection_name,
                    top_k=k,
                    include_metadata=True
                )
            )
        except Exception:
            return []

        # Convert to Documents
        documents = []
        for result in results:
            try:
                # Upstash returns a list of QueryResults with id, score, vector, and metadata
                metadata = result.metadata
                if metadata is None:
                    continue
                    
                content = metadata.get("content")
                if content is None:
                    continue
                    
                doc = Document(
                    page_content=content,
                    metadata={k: v for k, v in metadata.items() if k != "content"}
                )
                documents.append(doc)
            except Exception:
                continue

        return documents

def upstash_vec_factory(collection_name: str, embeddings, rest_url: str, rest_token: str, max_workers: int = 10) -> UpstashVectorStore:
    """Factory function to create Upstash vector store instances.
    
    Args:
        collection_name: Name of the collection
        embeddings: LangChain embeddings interface
        rest_url: Upstash REST URL
        rest_token: Upstash REST token
        max_workers: Maximum number of threads in the thread pool (default: 10)
    """
    return UpstashVectorStore(
        collection_name=collection_name,
        embedding_function=embeddings,
        rest_url=rest_url,
        rest_token=rest_token,
        max_workers=max_workers
    )
