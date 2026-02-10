"""Tests for pdf2anki.cache — extraction result caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf2anki.cache import (
    CacheEntry,
    _deserialize_entry,
    _serialize_entry,
    compute_file_hash,
    invalidate_cache,
    read_cache,
    write_cache,
)
from pdf2anki.extract import ExtractedDocument
from pdf2anki.section import Section

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> ExtractedDocument:
    """Minimal ExtractedDocument for cache tests."""
    return ExtractedDocument(
        source_path="test.txt",
        text="Hello world",
        chunks=("Hello world",),
        file_type="txt",
        used_ocr=False,
        sections=(),
    )


@pytest.fixture
def document_with_sections() -> ExtractedDocument:
    """ExtractedDocument with sections for round-trip tests."""
    return ExtractedDocument(
        source_path="doc.pdf",
        text="# Intro\nSome text\n# Chapter\nMore text",
        chunks=("# Intro\nSome text", "# Chapter\nMore text"),
        file_type="pdf",
        used_ocr=True,
        sections=(
            Section(
                id="section-0",
                heading="Intro",
                level=1,
                breadcrumb="doc > Intro",
                text="# Intro\nSome text",
                page_range="",
                char_count=18,
            ),
            Section(
                id="section-1",
                heading="Chapter",
                level=1,
                breadcrumb="doc > Chapter",
                text="# Chapter\nMore text",
                page_range="pp.1-3",
                char_count=20,
            ),
        ),
    )


@pytest.fixture
def sample_entry(sample_document: ExtractedDocument) -> CacheEntry:
    """Minimal CacheEntry for tests."""
    return CacheEntry(
        file_hash="abc123",
        source_path="test.txt",
        document=sample_document,
    )


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a temporary text file for hashing tests."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello world", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_returns_hex_string(self, sample_file: Path) -> None:
        result = compute_file_hash(sample_file)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex length

    def test_deterministic(self, sample_file: Path) -> None:
        h1 = compute_file_hash(sample_file)
        h2 = compute_file_hash(sample_file)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content", encoding="utf-8")
        f2.write_text("same content", encoding="utf-8")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "nonexistent.txt")

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_large_file_chunked(self, tmp_path: Path) -> None:
        """Verify large files are hashed correctly (chunked reading)."""
        f = tmp_path / "large.bin"
        content = b"x" * 200_000  # > _HASH_CHUNK_SIZE
        f.write_bytes(content)
        result = compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# _serialize_entry / _deserialize_entry
# ---------------------------------------------------------------------------


class TestSerializeDeserialize:
    def test_round_trip_simple(self, sample_entry: CacheEntry) -> None:
        data = _serialize_entry(sample_entry)
        restored = _deserialize_entry(data)
        assert restored.file_hash == sample_entry.file_hash
        assert restored.source_path == sample_entry.source_path
        assert restored.document.text == sample_entry.document.text
        assert restored.document.chunks == sample_entry.document.chunks
        assert restored.document.file_type == sample_entry.document.file_type
        assert restored.document.used_ocr == sample_entry.document.used_ocr
        assert restored.document.sections == sample_entry.document.sections

    def test_round_trip_with_sections(
        self, document_with_sections: ExtractedDocument
    ) -> None:
        entry = CacheEntry(
            file_hash="def456",
            source_path="doc.pdf",
            document=document_with_sections,
        )
        data = _serialize_entry(entry)
        restored = _deserialize_entry(data)
        assert len(restored.document.sections) == 2
        s0 = restored.document.sections[0]
        assert s0.id == "section-0"
        assert s0.heading == "Intro"
        assert s0.level == 1
        assert s0.breadcrumb == "doc > Intro"
        assert s0.page_range == ""
        assert s0.char_count == 18
        s1 = restored.document.sections[1]
        assert s1.page_range == "pp.1-3"

    def test_serialized_is_json_compatible(self, sample_entry: CacheEntry) -> None:
        data = _serialize_entry(sample_entry)
        json_str = json.dumps(data, ensure_ascii=False)
        reloaded = json.loads(json_str)
        restored = _deserialize_entry(reloaded)
        assert restored.document.text == sample_entry.document.text

    def test_deserialize_malformed_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            _deserialize_entry({"bad": "data"})


# ---------------------------------------------------------------------------
# write_cache / read_cache
# ---------------------------------------------------------------------------


class TestWriteReadCache:
    def test_write_creates_file(
        self, tmp_path: Path, sample_entry: CacheEntry
    ) -> None:
        cache_dir = tmp_path / "cache"
        path = write_cache(cache_dir, sample_entry)
        assert path.exists()
        assert path.suffix == ".json"

    def test_write_creates_directory(
        self, tmp_path: Path, sample_entry: CacheEntry
    ) -> None:
        cache_dir = tmp_path / "nested" / "cache"
        write_cache(cache_dir, sample_entry)
        assert cache_dir.is_dir()

    def test_read_after_write(
        self, tmp_path: Path, sample_entry: CacheEntry
    ) -> None:
        cache_dir = tmp_path / "cache"
        write_cache(cache_dir, sample_entry)
        result = read_cache(cache_dir, sample_entry.file_hash)
        assert result is not None
        assert result.file_hash == sample_entry.file_hash
        assert result.document.text == sample_entry.document.text

    def test_read_miss(self, tmp_path: Path) -> None:
        result = read_cache(tmp_path, "nonexistent_hash")
        assert result is None

    def test_read_nonexistent_dir(self, tmp_path: Path) -> None:
        result = read_cache(tmp_path / "no_such_dir", "abc123")
        assert result is None

    def test_read_corrupted_file(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        bad_file = cache_dir / "abc123.json"
        bad_file.write_text("not valid json{{{", encoding="utf-8")
        result = read_cache(cache_dir, "abc123")
        assert result is None

    def test_round_trip_with_sections(
        self, tmp_path: Path, document_with_sections: ExtractedDocument
    ) -> None:
        entry = CacheEntry(
            file_hash="sec456",
            source_path="doc.pdf",
            document=document_with_sections,
        )
        cache_dir = tmp_path / "cache"
        write_cache(cache_dir, entry)
        result = read_cache(cache_dir, "sec456")
        assert result is not None
        assert len(result.document.sections) == 2
        assert result.document.sections[0].heading == "Intro"
        assert result.document.used_ocr is True

    def test_overwrite_existing(
        self, tmp_path: Path, sample_document: ExtractedDocument
    ) -> None:
        cache_dir = tmp_path / "cache"
        entry1 = CacheEntry(
            file_hash="same_hash",
            source_path="v1.txt",
            document=sample_document,
        )
        write_cache(cache_dir, entry1)

        updated_doc = ExtractedDocument(
            source_path="v2.txt",
            text="Updated content",
            chunks=("Updated content",),
            file_type="txt",
            used_ocr=False,
            sections=(),
        )
        entry2 = CacheEntry(
            file_hash="same_hash",
            source_path="v2.txt",
            document=updated_doc,
        )
        write_cache(cache_dir, entry2)

        result = read_cache(cache_dir, "same_hash")
        assert result is not None
        assert result.document.text == "Updated content"


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------


class TestInvalidateCache:
    def test_invalidate_specific(
        self, tmp_path: Path, sample_entry: CacheEntry
    ) -> None:
        cache_dir = tmp_path / "cache"
        write_cache(cache_dir, sample_entry)
        count = invalidate_cache(cache_dir, sample_entry.file_hash)
        assert count == 1
        assert read_cache(cache_dir, sample_entry.file_hash) is None

    def test_invalidate_all(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        doc = ExtractedDocument(
            source_path="a.txt",
            text="aaa",
            chunks=("aaa",),
            file_type="txt",
            used_ocr=False,
        )
        for i in range(3):
            entry = CacheEntry(
                file_hash=f"hash_{i}",
                source_path=f"file_{i}.txt",
                document=doc,
            )
            write_cache(cache_dir, entry)

        count = invalidate_cache(cache_dir)
        assert count == 3
        assert list(cache_dir.glob("*.json")) == []

    def test_invalidate_nonexistent_hash(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        count = invalidate_cache(cache_dir, "nonexistent")
        assert count == 0

    def test_invalidate_nonexistent_dir(self, tmp_path: Path) -> None:
        count = invalidate_cache(tmp_path / "no_such_dir", "abc")
        assert count == 0


# ---------------------------------------------------------------------------
# CacheEntry immutability
# ---------------------------------------------------------------------------


class TestCacheEntryImmutability:
    def test_frozen(self, sample_entry: CacheEntry) -> None:
        with pytest.raises(AttributeError):
            sample_entry.file_hash = "new_hash"  # type: ignore[misc]

    def test_slots(self, sample_entry: CacheEntry) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sample_entry.extra = "value"  # type: ignore[attr-defined]
