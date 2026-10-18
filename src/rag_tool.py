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


async def rag_lookup(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.40,
) -> str:
    """
    Asynchronously retrieve relevant FAQ context for a given voice query.

    Encodes the query using 'paraphrase-multilingual-MiniLM-L12-v2' and searches
    the 'telecom_faq' collection in Qdrant, filtering results by `score_threshold`.

    Empirical Threshold Calibration:
        - EN query ('How do I pay my bill online?'): score ~0.75
        - ES query ('¿Cómo puedo pagar mi factura?'): score ~0.84
        - FR query ('Comment activer l'itinérance?'): score ~0.65
        - HI query ('मेरा इंटरनेट काम नहीं कर रहा है'): score ~0.44
        - Out-of-domain ('How to bake cookies'): score ~0.07
        Threshold 0.40 admits all 4 language inquiries while rejecting irrelevant topics.

    Args:
        query: Transcribed user speech or text question.
        top_k: Number of nearest neighbors to retrieve (default: 3).
        score_threshold: Minimum cosine similarity score required (default: 0.40).
            Queries scoring below this threshold return empty string, allowing the
            LLM to offer transfer to a human specialist rather than hallucinate.

    Returns:
        Concatenated text of top-matching FAQ records separated by double newlines,
        or an empty string if no relevant match passes the threshold.
    """
    if not query or not query.strip():
        return ""

    try:
        model = get_embedding_model()
        client = get_async_qdrant_client()

        # Run CPU-bound sentence encoding in worker thread to prevent event loop stutter.
        # PyTorch C++ kernels release the GIL during matrix multiplication.
        query_vector = await asyncio.to_thread(model.encode, query.strip())
        vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        # Search Qdrant vector database using query_points with score_threshold
        query_response = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector_list,
            limit=top_k,
            score_threshold=score_threshold,
        )

        points = getattr(query_response, "points", [])
        if not points:
            logger.info("No FAQ passages passed score_threshold (%.2f) for query: '%.40s...'", score_threshold, query)
            return ""

        extracted_chunks = [
            point.payload["text"]
            for point in points
            if point.payload and "text" in point.payload
        ]

        concatenated_context = "\n\n".join(extracted_chunks)
        logger.info(
            "rag_lookup retrieved %d passages (top score: %.3f) for query: '%.40s...'",
            len(extracted_chunks),
            points[0].score if points else 0.0,
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
