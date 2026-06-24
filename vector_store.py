from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import INDEX_DIR, EMBEDDING_MODEL
from ingest import load_and_chunk


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(force_rebuild: bool = False):
    embeddings = get_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    chunks = load_and_chunk()
    vector_store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


def get_retriever(force_rebuild: bool = False):
    vector_store = build_vector_store(force_rebuild=force_rebuild)
    from config import RETRIEVAL_K, MMR_FETCH_K, MMR_LAMBDA

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )