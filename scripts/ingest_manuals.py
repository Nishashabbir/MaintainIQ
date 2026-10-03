"""Run once after adding manuals, or whenever manuals change: python scripts/ingest_manuals.py"""

import json
import logging

from rag_system import ingest_documents

logging.basicConfig(level=logging.INFO)
print(json.dumps(ingest_documents(), indent=2))
