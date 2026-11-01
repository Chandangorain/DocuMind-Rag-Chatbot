import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from backend.config import (
    UPLOAD_DIR,
    DB_DIR,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_FILE_MB,
)


def get_collection_name(session_id: str) -> str:
    """One Chroma collection per session, so resumes never mix."""
    return f"resume_{session_id}"


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def ingest_pdf(file, session_id: str) -> int:
    """
    Takes an uploaded PDF (Streamlit UploadedFile), saves it,
    splits it into chunks, embeds them and stores them in Chroma.
    Returns the number of chunks stored.
    """
    filename = Path(file.name).name
    data = file.getvalue()

    # 1. Validate
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File is too large. Max size is {MAX_FILE_MB} MB.")

    # 2. Save to data/uploads/<session_id>/
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = session_dir / filename
    pdf_path.write_bytes(data)

    # 3. Load pages
    pages = PyPDFLoader(str(pdf_path)).load()
    if not any(p.page_content.strip() for p in pages):
        raise ValueError("No text found. This PDF may be scanned (image only).")

    # 4. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = filename  # show filename, not the full path

    # 5. Embed + store in this session's collection
    db = Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
    )
    db.add_documents(chunks)

    return len(chunks)


def clear_session(session_id: str) -> None:
    """Deletes the session's vectors and uploaded files (for 'Upload new resume')."""
    db = Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
    )
    db.delete_collection()
    shutil.rmtree(UPLOAD_DIR / session_id, ignore_errors=True)