"""Tests for license compliance — AGPL-3.0 regression guard.

Ensures that LICENSE file, pyproject.toml, README.md, README.ja.md,
and docs/RELEASE_RESEARCH.md all consistently reference AGPL-3.0.
Prevents accidental license regression.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================
# LICENSE file
# ============================================================


class TestLicenseFile:
    """Verify LICENSE file exists with correct AGPL-3.0 content."""

    def test_license_file_exists(self) -> None:
        license_path = PROJECT_ROOT / "LICENSE"
        assert license_path.exists(), "LICENSE file must exist at project root"

    def test_license_is_agpl3(self) -> None:
        text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
        assert "Version 3" in text

    def test_license_copyright_holder(self) -> None:
        text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "Copyright (C) 2026 shimo4228" in text


# ============================================================
# pyproject.toml
# ============================================================


class TestPyprojectLicense:
    """Verify pyproject.toml declares AGPL-3.0-or-later."""

    def test_license_field(self) -> None:
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'license = "AGPL-3.0-or-later"' in text


# ============================================================
# README files
# ============================================================


class TestReadmeLicense:
    """Verify README.md and README.ja.md reference AGPL-3.0."""

    def test_readme_en(self) -> None:
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "AGPL-3.0 License" in text
        assert "[LICENSE](LICENSE)" in text

    def test_readme_ja(self) -> None:
        text = (PROJECT_ROOT / "README.ja.md").read_text(encoding="utf-8")
        assert "AGPL-3.0 License" in text
        assert "[LICENSE](LICENSE)" in text


# ============================================================
# Release research doc
# ============================================================


class TestReleaseResearch:
    """Verify RELEASE_RESEARCH.md marks license issue as resolved."""

    def test_license_resolved(self) -> None:
        text = (PROJECT_ROOT / "docs" / "RELEASE_RESEARCH.md").read_text(
            encoding="utf-8",
        )
        assert "解決済み（AGPL-3.0 に変更）" in text

    def test_action_item_checked(self) -> None:
        text = (PROJECT_ROOT / "docs" / "RELEASE_RESEARCH.md").read_text(
            encoding="utf-8",
        )
        assert "[x] **ライセンス問題を解決する**" in text
