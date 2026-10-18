"""
Asynchronous semantic retrieval tool for VoiceRAG Telephony.

Provides `rag_lookup`, a low-latency cross-lingual lookup utility that retrieves
relevant telecom FAQ passages from the Qdrant vector database using
'paraphrase-multilingual-MiniLM-L12-v2' embeddings.
"""

import asyncio
import logging
import os
from pathlib import Path
import threading
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voicerag.rag_tool")

# Resolve project paths and load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

COLLECTION_NAME = "telecom_faq"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QDRANT_URL = "http://localhost:6333"

# Module-level singletons (lazy initialization)
_embedding_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

_async_client: Optional[AsyncQdrantClient] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy singleton accessor for the SentenceTransformer embedding model.
    Guarantees that the model is loaded into memory only once.
    """
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                logger.info("Lazy-loading embedding model: %s", EMBEDDING_MODEL_NAME)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_async_qdrant_client() -> AsyncQdrantClient:
    """
    Lazy singleton accessor for AsyncQdrantClient configured via environment variables.
    """
    global _async_client
    if _async_client is None:
        url = os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        api_key = os.getenv("QDRANT_API_KEY") or None
        logger.info("Initializing AsyncQdrantClient connecting to: %s", url)
        _async_client = AsyncQdrantClient(url=url, api_key=api_key)
    return _async_client


async def rag_lookup(query: str, top_k: int = 3) -> str:
    """
    Asynchronously retrieve relevant FAQ context for a given voice query.

    Encodes the query using 'paraphrase-multilingual-MiniLM-L12-v2' and searches
    the 'telecom_faq' collection in Qdrant, returning the concatenated text of the top_k results.

    Supports cross-lingual mapping (e.g. Hindi, Spanish, French queries match English documents).

    Args:
        query: Transcribed user speech or text question.
        top_k: Number of nearest neighbors to retrieve (default: 3).

    Returns:
        Concatenated text of top-matching FAQ records separated by double newlines,
        or an empty string if no relevant match is found or an error occurs.
    """
    if not query or not query.strip():
        return ""

    try:
        model = get_embedding_model()
        client = get_async_qdrant_client()

        # Run CPU-bound sentence encoding in worker thread to prevent event loop stutter
        query_vector = await asyncio.to_thread(model.encode, query.strip())
        vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        # Search the Qdrant vector database asynchronously
        search_results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector_list,
            limit=top_k,
        )

        if not search_results:
            logger.debug("No search results found for query: %s", query)
            return ""

        extracted_chunks = [
            hit.payload["text"]
            for hit in search_results
            if hit.payload and "text" in hit.payload
        ]

        concatenated_context = "\n\n".join(extracted_chunks)
        logger.info(
            "rag_lookup retrieved %d passages for query: '%.40s...'",
            len(extracted_chunks),
            query,
        )
        return concatenated_context

    except Exception as exc:
        logger.error("Error executing rag_lookup for query '%s': %s", query, exc, exc_info=True)
        return ""


if __name__ == "__main__":
    async def _test():
        test_queries = [
            "How do I pay my bill?",
            "¿Cómo puedo pagar mi factura telefónica?",
            "मेरा इंटरनेट काम नहीं कर रहा है",
            "Comment activer l'itinérance internationale ?"
        ]
        print("Testing rag_lookup with sample queries...")
        for q in test_queries:
            print(f"\n[Query]: {q}")
            res = await rag_lookup(q, top_k=2)
            print(f"[Result]:\n{res if res else '(No results or Qdrant offline)'}")

    asyncio.run(_test())
