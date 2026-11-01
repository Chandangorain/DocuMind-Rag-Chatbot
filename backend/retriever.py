from langchain_chroma import Chroma

from backend.config import DB_DIR, TOP_K
from backend.ingest import get_collection_name, get_embeddings


def get_retriever(session_id: str):
    """
    Opens the session's Chroma collection (created by ingest_pdf)
    and returns a retriever that fetches the TOP_K most similar chunks.
    """
    db = Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
    )
    return db.as_retriever(search_kwargs={"k": TOP_K})