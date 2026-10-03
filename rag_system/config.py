"""Configuration for the standalone MaintainIQ RAG module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = PROJECT_ROOT / "data" / "manuals"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    manifest_path: Path = PROJECT_ROOT / "data" / "rag_manifest.json"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_relevance: float = 0.20

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            documents_dir=Path(os.getenv("RAG_DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "manuals"))),
            chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))),
            manifest_path=Path(os.getenv("RAG_MANIFEST_PATH", str(PROJECT_ROOT / "data" / "rag_manifest.json"))),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 150),
            top_k=_env_int("RAG_TOP_K", 5),
            min_relevance=float(os.getenv("RAG_MIN_RELEVANCE", "0.20")),
        )


settings = Settings.from_env()
