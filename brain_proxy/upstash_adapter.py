from typing import List
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.upstash import UpstashVectorStore as LangchainUpstashVectorStore
import asyncio
from concurrent.futures import ThreadPoolExecutor


class UpstashAsyncWrapper:
    """Async wrapper for Upstash Vector to maintain consistency with ChromaDB adapter."""
    
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
            max_workers: Maximum number of threads in the thread pool (default: 10)
        """
        # Ensure URL has protocol
        if not rest_url.startswith(('http://', 'https://')):
            rest_url = f'https://{rest_url}'
            
        self.upstash = LangchainUpstashVectorStore(
            embedding=embedding_function,
            index_url=rest_url,
            index_token=rest_token,
            namespace=collection_name,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Converts LangChain Documents to Upstash format by creating new Documents with text field.
        """
        if not documents:
            return

        # Convert documents to Upstash format
        upstash_docs = []
        for doc in documents:
            # Create a new Document with both text and page_content fields
            upstash_doc = Document(
                page_content=doc.page_content,
                metadata={
                    'text': doc.page_content,  # Add text field for Upstash
                    **doc.metadata  # Preserve any metadata
                }
            )
            upstash_docs.append(upstash_doc)
            
        await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.upstash.add_documents,
            upstash_docs
        )

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Returns LangChain Documents directly from Upstash.
        """
        # Upstash already returns LangChain Documents
        documents = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self.upstash.similarity_search,
            query,
            k
        )
        return documents


def upstash_vec_factory(collection_name: str, embeddings, rest_url: str, rest_token: str, max_workers: int = 10) -> UpstashAsyncWrapper:
    """Factory function to create Upstash vector store instances.
    
    Args:
        collection_name: Name of the collection
        embeddings: LangChain embeddings interface
        rest_url: Upstash REST URL
        rest_token: Upstash REST token
        max_workers: Maximum number of threads in the thread pool (default: 10)
    """
    return UpstashAsyncWrapper(
        collection_name=collection_name,
        embedding_function=embeddings,
        rest_url=rest_url,
        rest_token=rest_token,
        max_workers=max_workers
    )
