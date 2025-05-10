from typing import List, Optional
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from upstash_vector import Index
from datetime import datetime, timezone
import asyncio
from functools import partial

class UpstashVectorStore:
    """Upstash vector store adapter for brain-proxy."""

    def __init__(
        self,
        collection_name: str,
        embedding_function: Embeddings,
        rest_url: str,
        rest_token: str,
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
        self.index = Index(url=rest_url, token=rest_token)

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
        loop = asyncio.get_event_loop()
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            # Run synchronous upsert in a thread pool
            await loop.run_in_executor(
                None,
                partial(
                    self.index.upsert,
                    vectors=batch,
                    namespace=self.collection_name
                )
            )

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        # Get query embedding
        query_embedding = await self.embedding_function.aembed_query(query)

        # Search Upstash (run sync query in thread pool)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            partial(
                self.index.query,
                vector=query_embedding,
                namespace=self.collection_name,
                top_k=k
            )
        )

        # Convert to Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.metadata["content"],
                metadata={k: v for k, v in result.metadata.items() if k != "content"}
            )
            documents.append(doc)

        return documents

def upstash_vec_factory(collection_name: str, embeddings, rest_url: str, rest_token: str) -> UpstashVectorStore:
    """Factory function to create Upstash vector store instances."""
    return UpstashVectorStore(
        collection_name=collection_name,
        embedding_function=embeddings,
        rest_url=rest_url,
        rest_token=rest_token
    )
