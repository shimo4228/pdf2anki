"""Extraction cache for pdf2anki.

Caches text extraction results (ExtractedDocument) to avoid redundant
PDF/TXT/MD parsing on repeated runs. Cache key is SHA-256 of file content.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdf2anki.extract import ExtractedDocument
from pdf2anki.section import Section

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 65536  # 64 KB reads for large-file hashing


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Cached extraction result with metadata."""

    file_hash: str
    source_path: str
    document: ExtractedDocument


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents using chunked reads.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def read_cache(cache_dir: Path, file_hash: str) -> CacheEntry | None:
    """Read a cached extraction result.

    Args:
        cache_dir: Directory containing cache files.
        file_hash: SHA-256 hash of the source file.

    Returns:
        CacheEntry if cache hit and valid, None otherwise.
    """
    cache_file = cache_dir / f"{file_hash}.json"
    if not cache_file.is_file():
        return None

    try:
        raw = cache_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        return _deserialize_entry(data)
    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("Corrupted cache entry: %s", cache_file)
        return None


def write_cache(cache_dir: Path, entry: CacheEntry) -> Path:
    """Write an extraction result to the cache.

    Creates cache_dir if it does not exist.

    Args:
        cache_dir: Directory to store cache files.
        entry: Cache entry to persist.

    Returns:
        Path to the written cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{entry.file_hash}.json"
    data = _serialize_entry(entry)
    cache_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cache_file


def invalidate_cache(cache_dir: Path, file_hash: str | None = None) -> int:
    """Remove cache entries.

    Args:
        cache_dir: Directory containing cache files.
        file_hash: Specific hash to invalidate, or None for all entries.

    Returns:
        Number of cache entries removed.
    """
    if not cache_dir.is_dir():
        return 0

    if file_hash is not None:
        cache_file = cache_dir / f"{file_hash}.json"
        if cache_file.is_file():
            cache_file.unlink()
            return 1
        return 0

    files = list(cache_dir.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)


def _serialize_entry(entry: CacheEntry) -> dict[str, Any]:
    """Serialize a CacheEntry to a JSON-compatible dict."""
    doc = entry.document
    sections_data = [
        {
            "id": s.id,
            "heading": s.heading,
            "level": s.level,
            "breadcrumb": s.breadcrumb,
            "text": s.text,
            "page_range": s.page_range,
            "char_count": s.char_count,
        }
        for s in doc.sections
    ]
    return {
        "file_hash": entry.file_hash,
        "source_path": entry.source_path,
        "document": {
            "source_path": doc.source_path,
            "text": doc.text,
            "chunks": list(doc.chunks),
            "file_type": doc.file_type,
            "used_ocr": doc.used_ocr,
            "sections": sections_data,
        },
    }


def _deserialize_entry(data: dict[str, Any]) -> CacheEntry:
    """Deserialize a dict to a CacheEntry.

    Raises:
        ValueError: If data is malformed.
        KeyError: If required keys are missing.
    """
    doc_data = data["document"]
    sections = tuple(
        Section(
            id=s["id"],
            heading=s["heading"],
            level=s["level"],
            breadcrumb=s["breadcrumb"],
            text=s["text"],
            page_range=s["page_range"],
            char_count=s["char_count"],
        )
        for s in doc_data.get("sections", [])
    )
    document = ExtractedDocument(
        source_path=doc_data["source_path"],
        text=doc_data["text"],
        chunks=tuple(doc_data["chunks"]),
        file_type=doc_data["file_type"],
        used_ocr=doc_data["used_ocr"],
        sections=sections,
    )
    return CacheEntry(
        file_hash=data["file_hash"],
        source_path=data["source_path"],
        document=document,
    )
