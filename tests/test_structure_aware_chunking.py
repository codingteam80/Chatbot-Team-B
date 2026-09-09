import tempfile
from pathlib import Path

import fitz

from ingestion.metadata import MetadataBuilder
from ingestion.pdf_structure import PDFStructureExtractor
from config.settings import INDEX_SCHEMA_VERSION
from utils.manifest import ManifestManager


def _make_pdf(path: Path):
    doc = fitz.open()

    for page_number in (1, 2, 3):
        page = doc.new_page()
        page.insert_text((72, 30), "Company Policy Manual", fontsize=8)
        page.insert_text((72, 805), f"Page {page_number} of 3", fontsize=8)

        if page_number == 1:
            page.insert_text((72, 90), "SECTION 1 Purpose", fontsize=16)
            page.insert_text(
                (72, 125),
                "This policy explains the purpose and scope for all employees.",
                fontsize=10,
            )
        elif page_number == 2:
            page.insert_text((72, 90), "Rule 8.4", fontsize=15)
            page.insert_text(
                (72, 125),
                "Employees must follow the documented approval workflow before travel.",
                fontsize=10,
            )
            # A body-text cross-reference must not become a fake rule heading.
            page.insert_text((72, 160), "Rule 9.9", fontsize=10)
            page.insert_text(
                (72, 180),
                "This is only a cross-reference in normal body text.",
                fontsize=10,
            )
        else:
            page.insert_text((72, 90), "2 Eligibility", fontsize=14)
            page.insert_text(
                (72, 125),
                "Regular employees are eligible after completing the required service period.",
                fontsize=10,
            )

    doc.save(path)
    doc.close()


def test_pdf_structure_detects_sections_and_removes_repeated_chrome():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.pdf"
        _make_pdf(path)

        sections = PDFStructureExtractor.extract(str(path))
        combined = "\n".join(section.text for section in sections)

        assert "SECTION 1 Purpose" in combined
        assert "Rule 8.4" in combined
        assert "Company Policy Manual" not in combined
        assert "Page 1 of 3" not in combined
        assert sum(section.rule_id == "8.4" for section in sections) == 1
        assert not any(section.rule_id == "9.9" for section in sections)
        assert any(section.section_title == "2 Eligibility" for section in sections)


def test_metadata_keeps_structure_fields_chroma_safe():
    metadata = MetadataBuilder.build(
        file_path="policy.pdf",
        chunk_id=0,
        total_chunks=2,
        extra_metadata={
            "parser_strategy": "pdf_structure_aware",
            "page_start": 4,
            "page_end": 5,
            "section_title": "Rule 8.4",
            "empty_value": None,
        },
    )

    assert metadata["parser_strategy"] == "pdf_structure_aware"
    assert metadata["page_start"] == 4
    assert metadata["page_end"] == 5
    assert "empty_value" not in metadata
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())


def test_manifest_marks_current_index_schema_version():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text("sample company knowledge", encoding="utf-8")

        manifest = ManifestManager.build([path])
        info = next(iter(manifest.values()))

        assert info["index_schema_version"] == INDEX_SCHEMA_VERSION


def test_schema_upgrade_changes_candidate_index_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text("same file contents", encoding="utf-8")

        baseline = ManifestManager.build([path])
        key = next(iter(baseline))
        old = dict(baseline)
        old[key] = dict(old[key])
        old[key]["index_schema_version"] = "legacy"
        old[key]["last_indexed"] = "2000-01-01 00:00:00"

        upgraded = ManifestManager.build([path], previous_manifest=old)

        assert upgraded[key]["index_schema_version"] == INDEX_SCHEMA_VERSION
        assert upgraded[key]["last_indexed"] != "2000-01-01 00:00:00"
