from docx import Document

from ingestion.loaders.base_loader import BaseLoader
from ingestion.parsed_document import ParsedDocument


class DOCXLoader(BaseLoader):
    """Optional DOCX loader kept import-safe for future structured routing."""

    def load(self, file_path):
        document = Document(file_path)
        blocks = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                blocks.append(text)

        text = "\n".join(blocks).strip()
        if not text:
            return []

        return [ParsedDocument(text=text)]
