from __future__ import annotations

from pathlib import Path

# Native loaders used first for common office/engineering files.  Keeping these
# imports local to Python packages already in requirements.txt lets a normal
# technical KB rebuild proceed even if the optional ``unstructured`` package is
# missing or temporarily broken on one workstation.
from ingestion.loaders.txt_loader import TXTLoader
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.json_loader import JSONLoader
from ingestion.loaders.xml_loader import XMLLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.xlsx_loader import XLSXLoader
from ingestion.loaders.pptx_loader import PPTXLoader


CUSTOM_LOADERS = {
    ".txt": TXTLoader(),
    ".csv": CSVLoader(),
    ".json": JSONLoader(),
    ".xml": XMLLoader(),
    ".pdf": PDFLoader(),
    ".docx": DOCXLoader(),
    ".xlsx": XLSXLoader(),
    ".pptx": PPTXLoader(),
}


class DocumentParser:
    """Parse one supported document into plain text.

    ``unstructured`` is intentionally imported only for extensions that do not
    have a native DocuBot loader.  This prevents the Update Knowledge Base
    button from crashing at import time when the current corpus contains only
    PDFs/Office formats that DocuBot can already parse natively.
    """

    @staticmethod
    def _flatten_loaded(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                text = getattr(item, "text", item)
                clean = str(text or "").strip()
                if clean:
                    parts.append(clean)
            return "\n\n".join(parts).strip()

        return str(value).strip()

    @staticmethod
    def _parse_with_unstructured(file_path: str) -> str:
        try:
            from unstructured.partition.auto import partition
        except ImportError as error:
            extension = Path(file_path).suffix.lower() or "this file type"
            raise RuntimeError(
                "Optional document parser dependency 'unstructured' is not "
                f"installed for {extension}. Run "
                "Setup_DocuBot_Production_Environment.bat, then retry the KB update."
            ) from error

        elements = partition(filename=file_path)
        return "\n".join(
            str(element).strip()
            for element in elements
            if str(element).strip()
        ).strip()

    @staticmethod
    def parse(file_path: str) -> str:
        path = Path(file_path)
        extension = path.suffix.lower()

        try:
            loader = CUSTOM_LOADERS.get(extension)
            if loader is not None:
                print(f"[NATIVE LOADER] {extension} -> {file_path}")
                text = DocumentParser._flatten_loaded(loader.load(file_path))
            else:
                print(f"[UNSTRUCTURED] {extension} -> {file_path}")
                text = DocumentParser._parse_with_unstructured(file_path)

            if not text.strip():
                raise RuntimeError("Parser returned no usable text.")

            return text.strip()

        except Exception as error:
            # Propagate parsing failures so the transactional rebuild aborts and
            # the previous working Qdrant/BM25 state is preserved.  Silently
            # returning an empty document can otherwise publish an incomplete KB.
            print(f"[PARSER ERROR] {file_path}: {type(error).__name__}: {error}")
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(
                f"Could not parse '{path.name}': {type(error).__name__}: {error}"
            ) from error
