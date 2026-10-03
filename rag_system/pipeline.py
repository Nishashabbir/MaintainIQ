"""Question answering over indexed maintenance manuals."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from .config import settings
from .ingestion import COLLECTION_NAME, _get_collection, ingest_documents

logger = logging.getLogger(__name__)
_embedding_model: SentenceTransformer | None = None

NO_EVIDENCE_ANSWER = (
    "I could not find enough relevant information in the available maintenance documents "
    "to answer this reliably. Please consult the equipment manual or a qualified technician."
)


def _model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _empty_response(message: str = NO_EVIDENCE_ANSWER, *, error: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "answer": message,
        "sources": [],
        "supporting_passages": [],
        "grounded": False,
    }
    if error:
        response["error"] = error
    return response


def ask_maintenance_question(question: str) -> dict[str, Any]:
    """Answer from manual evidence; citations include document, page and passage.

    Returns a stable dictionary suitable for calling from MaintainIQ's Knowledge Agent.
    """
    if not isinstance(question, str) or not question.strip():
        return _empty_response("Please provide a maintenance question.", error="empty_question")
    if not __import__("os").getenv("GROQ_API_KEY"):
        return _empty_response("RAG is not configured because the GROQ_API_KEY is missing.", error="missing_api_key")

    try:
        collection = _get_collection(settings)
        if collection.count() == 0:
            ingest_result = ingest_documents()
            if collection.count() == 0:
                detail = "; ".join(ingest_result["errors"])
                return _empty_response(
                    "No readable maintenance manual content is indexed yet." + (f" {detail}" if detail else ""),
                    error="no_indexed_documents",
                )

        query_embedding = _model().encode([question.strip()], normalize_embeddings=True).tolist()[0]
        found = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(settings.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = (found.get("documents") or [[]])[0]
        metas = (found.get("metadatas") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        passages: list[dict[str, Any]] = []
        for text, metadata, distance in zip(docs, metas, distances):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            if relevance < settings.min_relevance:
                continue
            passages.append({
                "document": metadata.get("source", "Unknown document"),
                "page": metadata.get("page"),
                "text": text,
                "relevance": round(relevance, 4),
            })
        if not passages:
            return _empty_response()

        context = "\n\n".join(
            f"[Source {i}: {p['document']}, page {p['page']}]\n{p['text']}"
            for i, p in enumerate(passages, start=1)
        )
        client = Groq()
        completion = client.chat.completions.create(
            model=settings.groq_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MaintainIQ's maintenance manual assistant. Answer only with facts and "
                        "instructions supported by the supplied excerpts. Do not use outside knowledge "
                        "or invent steps, limits, or safety guidance. If the excerpts do not adequately "
                        "answer the question, say that they do not provide enough information. Cite "
                        "claims using [Source N]. Preserve warnings and conditions from the excerpts."
                    ),
                },
                {"role": "user", "content": f"Question: {question.strip()}\n\nManual excerpts:\n{context}"},
            ],
        )
        answer = completion.choices[0].message.content or NO_EVIDENCE_ANSWER
        insufficient = any(phrase in answer.lower() for phrase in (
            "not provide enough information", "not enough information", "cannot determine",
            "could not find enough", "do not contain enough",
        ))
        if insufficient:
            return _empty_response(answer)
        source_list = [{"document": p["document"], "page": p["page"]} for p in passages]
        unique_sources = list({(item["document"], item["page"]): item for item in source_list}.values())
        return {
            "answer": answer,
            "sources": unique_sources,
            "supporting_passages": passages,
            "grounded": True,
        }
    except Exception as exc:
        logger.exception("Maintenance RAG query failed")
        return _empty_response(
            "The maintenance document search is temporarily unavailable. Please try again or consult a qualified technician.",
            error=f"retrieval_error: {exc}",
        )
