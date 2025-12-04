import os
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from fastapi import Depends
from dotenv import load_dotenv

load_dotenv()

_client: ClientAPI | None = None
_collection: Collection | None = None

# Embeddings con OpenAI (sin sentence-transformers ni torch)
_embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large",  # o "text-embedding-3-small" si prefieres
)

def get_chroma_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
    return _client

def get_chroma_collection(
    client: ClientAPI = Depends(get_chroma_client),
) -> Collection:
    global _collection
    if _collection is None:
        _collection = client.get_or_create_collection(
            name="mis_documentos_openai",
            embedding_function=_embedder,
        )

    return _collection
