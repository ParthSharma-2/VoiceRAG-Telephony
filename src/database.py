"""
Database provisioning and vector seeding module for VoiceRAG Telephony.

This module initializes the Qdrant vector database collection 'telecom_faq',
chunks the multilingual telecom FAQ knowledge base, generates dense vector
embeddings using 'paraphrase-multilingual-MiniLM-L12-v2', and upserts the records.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voicerag.database")

# Resolve project paths relative to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "faq_knowledge_base.txt"
ENV_PATH = PROJECT_ROOT / ".env"

# Load environment configuration from .env if present
load_dotenv(dotenv_path=ENV_PATH)

COLLECTION_NAME = "telecom_faq"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
DEFAULT_QDRANT_URL = "http://localhost:6333"


def get_qdrant_client(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
) -> QdrantClient:
    """
    Initialize and return a synchronous QdrantClient.
    
    Args:
        qdrant_url: Optional endpoint URL. Defaults to env var QDRANT_URL or localhost.
        qdrant_api_key: Optional API key. Defaults to env var QDRANT_API_KEY.

    Returns:
        Configured QdrantClient instance.
    """
    url = qdrant_url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
    api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY") or None
    logger.info("Initializing QdrantClient connecting to: %s", url)
    return QdrantClient(url=url, api_key=api_key)


def load_and_chunk_knowledge_base(
    faq_path: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Reads the FAQ knowledge base file and splits it into discrete chunks.

    Args:
        faq_path: Path to the FAQ text file. Defaults to data/faq_knowledge_base.txt.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlap character count between adjacent chunks.

    Returns:
        List of cleaned text chunks.
    """
    target_path = Path(faq_path) if faq_path else DEFAULT_FAQ_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"FAQ knowledge base file not found at: {target_path}")

    logger.info("Reading knowledge base from: %s", target_path)
    with open(target_path, "r", encoding="utf-8") as f:
        faq_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(faq_text)
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    logger.info("Extracted %d chunks from FAQ document", len(cleaned_chunks))
    return cleaned_chunks


def seed_database(
    faq_path: Optional[Path] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Provisions the Qdrant collection and embeds/upserts all chunks from the FAQ knowledge base.

    Args:
        faq_path: Optional custom path to the FAQ knowledge base text.
        qdrant_url: Qdrant endpoint URL (falls back to env or http://localhost:6333).
        qdrant_api_key: Optional Qdrant API key.
        collection_name: Target vector collection name (defaults to 'telecom_faq').

    Returns:
        The total number of vector points successfully upserted.
    """
    # 1. Load and chunk text
    chunks = load_and_chunk_knowledge_base(faq_path=faq_path)
    if not chunks:
        logger.warning("No text chunks were extracted. Seeding aborted.")
        return 0

    # 2. Initialize Qdrant Client
    client = get_qdrant_client(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)

    # 3. Create or recreate the collection with size=384 and Distance.COSINE
    try:
        if client.collection_exists(collection_name=collection_name):
            logger.info("Collection '%s' already exists. Deleting prior collection...", collection_name)
            client.delete_collection(collection_name=collection_name)

        logger.info(
            "Creating collection '%s' with vector size=%d and distance=COSINE",
            collection_name,
            VECTOR_SIZE,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception as exc:
        logger.warning("Standard collection provisioning encountered: %s. Attempting recreate_collection fallback...", exc)
        if hasattr(client, "recreate_collection"):
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        else:
            raise

    # 4. Generate Embeddings using SentenceTransformer
    logger.info("Loading sentence transformer model '%s'...", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Computing dense embeddings for %d text chunks...", len(chunks))
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # 5. Build and upsert points
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            payload={"text": chunk, "chunk_id": idx, "source": "faq_knowledge_base.txt"},
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    logger.info("Upserting %d points into collection '%s'...", len(points), collection_name)
    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    logger.info("Successfully seeded database with %d chunks.", len(points))
    return len(points)


if __name__ == "__main__":
    print("Starting Qdrant seeding routine...")
    try:
        count = seed_database()
        print(f"Database seeding completed successfully! Upserted {count} records.")
    except Exception as e:
        logger.error("Failed to seed database: %s", e, exc_info=True)
        raise SystemExit(1)
