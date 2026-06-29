import fitz

from ingestion.document import ParsedDocument
from ingestion.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, file_path):

        pdf = fitz.open(file_path)

        documents = []

        for page_index in range(len(pdf)):

            page = pdf[page_index]

            text = page.get_text()

            text = text.strip()

            if not text:
                continue

            documents.append(

                ParsedDocument(
                    text=text,
                    page_number=page_index + 1
                )

            )

        print(
            f"[PDF] Pages created: {len(documents)}"
        )

        return documents