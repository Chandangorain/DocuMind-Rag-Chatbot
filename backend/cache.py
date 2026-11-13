import json
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import DB_DIR, CACHE_DISTANCE_THRESHOLD
from backend.ingest import get_embeddings

# Words that suggest the question depends on earlier chat
FOLLOW_UP_WORDS = {
    "it", "that", "this", "those", "these", "them", "they",
    "previous", "above", "earlier", "same",
    "first", "second", "third", "last", "another", "also", "more",
}
FOLLOW_UP_STARTS = ("and ", "also ", "what about", "how about")


def get_cache_name(session_id: str) -> str:
    """One cache collection per session, separate from the document collection."""
    return f"cache_{session_id}"


def _open_cache(session_id: str):
    return Chroma(
        collection_name=get_cache_name(session_id),
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},   # distance = 1 - similarity
    )


def looks_like_follow_up(question: str, has_history: bool) -> bool:
    """True if the answer probably depends on earlier messages."""
    if not has_history:
        return False
    q = question.lower().strip()
    words = set(re.findall(r"[a-z']+", q))
    return bool(words & FOLLOW_UP_WORDS) or q.startswith(FOLLOW_UP_STARTS)


def lookup(session_id: str, question: str):
    """Returns (answer, sources) if a similar question was answered before, else None."""
    db = _open_cache(session_id)
    results = db.similarity_search_with_score(question, k=1)
    if not results:
        return None
    doc, distance = results[0]
    if distance <= CACHE_DISTANCE_THRESHOLD:
        return doc.metadata["answer"], json.loads(doc.metadata["sources"])
    return None


def store(session_id: str, question: str, answer: str, sources: list) -> None:
    """Saves a question with its answer. The question text is what gets embedded."""
    db = _open_cache(session_id)
    db.add_documents([
        Document(
            page_content=question,
            metadata={"answer": answer, "sources": json.dumps(sources)},
        )
    ])


def clear_cache(session_id: str) -> None:
    """Deletes this session's cache (used when the documents change)."""
    try:
        _open_cache(session_id).delete_collection()
    except Exception:
        pass