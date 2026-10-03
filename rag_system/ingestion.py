"""PDF extraction and persistent, incremental Chroma indexing."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import Settings, settings

logger = logging.getLogger(__name__)
COLLECTION_NAME = "maintainiq_manuals"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": page_index + 1})
    return pages


def _get_collection(cfg: Settings):
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "MaintainIQ maintenance manual chunks"},
    )


def ingest_documents(
    documents_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Index new or changed PDFs and remove chunks for deleted PDFs."""
    cfg = settings
    root = Path(documents_dir) if documents_dir is not None else cfg.documents_dir
    result: dict[str, Any] = {"indexed": [], "skipped": [], "removed": [], "errors": []}
    if not root.exists():
        result["errors"].append(f"Manuals directory does not exist: {root}")
        return result

    pdfs = sorted(root.rglob("*.pdf"))
    collection = _get_collection(cfg)
    manifest = _load_manifest(cfg.manifest_path)
    current_names = {str(path.resolve()) for path in pdfs}

    for deleted in set(manifest) - current_names:
        collection.delete(where={"source_path": deleted})
        manifest.pop(deleted, None)
        result["removed"].append(deleted)

    pending: list[tuple[Path, str, list[dict[str, Any]]]] = []
    for path in pdfs:
        key = str(path.resolve())
        try:
            digest = _sha256(path)
            if not force and manifest.get(key) == digest:
                result["skipped"].append(path.name)
                continue
            pages = _extract_pages(path)
            if not pages:
                result["errors"].append(f"No extractable text in {path.name}; skipped.")
                continue
            pending.append((path, digest, pages))
        except Exception as exc:  # Keep indexing other manuals when one PDF fails.
            logger.exception("Could not extract %s", path)
            result["errors"].append(f"Could not process {path.name}: {exc}")

    if pending:
        model = SentenceTransformer(cfg.embedding_model)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=min(cfg.chunk_overlap, max(0, cfg.chunk_size - 1)),
            add_start_index=True,
        )
        for path, digest, pages in pending:
            key = str(path.resolve())
            try:
                chunks: list[dict[str, Any]] = []
                for page in pages:
                    for chunk in splitter.create_documents(
                        [page["text"]], metadatas=[{"page": page["page"]}]
                    ):
                        chunks.append({
                            "text": chunk.page_content,
                            "page": int(chunk.metadata["page"]),
                            "start_index": int(chunk.metadata.get("start_index", 0)),
                        })
                if not chunks:
                    result["errors"].append(f"No usable text chunks in {path.name}; skipped.")
                    continue

                # Delete old versions only after the replacement chunks are ready.
                ids = [hashlib.sha256(f"{key}:{digest}:{i}".encode()).hexdigest() for i in range(len(chunks))]
                embeddings = model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True).tolist()
                collection.delete(where={"source_path": key})
                collection.add(
                    ids=ids,
                    documents=[chunk["text"] for chunk in chunks],
                    embeddings=embeddings,
                    metadatas=[{
                        "source": path.name,
                        "source_path": key,
                        "page": chunk["page"],
                        "start_index": chunk["start_index"],
                        "content_hash": digest,
                    } for chunk in chunks],
                )
                manifest[key] = digest
                result["indexed"].append({"document": path.name, "chunks": len(chunks)})
            except Exception as exc:
                logger.exception("Could not index %s", path)
                result["errors"].append(f"Could not index {path.name}: {exc}")

    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
