# MaintainIQ RAG System

Standalone retrieval-augmented question answering over maintenance PDF manuals. The main application can import `ask_maintenance_question` without importing a UI or agent framework.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-rag.txt
Copy-Item .env.example .env
```

Add the Groq API key to `.env`, then put PDF manuals under `data/manuals/` (nested folders are supported). Run `python scripts/ingest_manuals.py` to index them. The persistent Chroma store is under `data/chroma/`; the manifest tracks SHA-256 digests so unchanged files are skipped. Changed files are re-indexed and removed files are removed from the collection.

## Main application interface

```python
from rag_system import ask_maintenance_question

result = ask_maintenance_question("What should I check if CNC-001 is overheating?")
print(result["answer"])
for source in result["sources"]:
    print(source["document"], source["page"])
```

The result includes `answer`, structured `sources` (`document`, `page`), `supporting_passages` (`document`, `page`, `text`, `relevance`), and `grounded`. On errors or insufficient evidence it returns `grounded: false`, an explanatory answer, and, where applicable, an `error` code.

The first ingestion downloads/loads the embedding model and may take a few minutes. Image-only scanned PDFs need OCR before they contain searchable text; the ingestion report marks PDFs with no extractable text. To re-index all manuals explicitly, call `ingest_documents(force=True)`.
