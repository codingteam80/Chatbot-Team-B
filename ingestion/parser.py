from pathlib import Path

# Universal document parser (Third-party library)
from unstructured.partition.auto import partition

# Custom loaders
from ingestion.loaders.txt_loader import TXTLoader
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.json_loader import JSONLoader
from ingestion.loaders.xml_loader import XMLLoader

#from ingestion.loaders.pdf_loader import PDFLoader
#from ingestion.loaders.docx_loader import DOCXLoader
#from ingestion.loaders.xlsx_loader import XLSXLoader
#from ingestion.loaders.pptx_loader import PPTXLoader

# Custom loaders for structured documents
CUSTOM_LOADERS = {
    ".txt": TXTLoader(),
    ".csv": CSVLoader(),
    ".json": JSONLoader(),
    ".xml": XMLLoader(),

#    ".pdf": PDFLoader(),
#    ".docx": DOCXLoader(),
#    ".xlsx": XLSXLoader(),
#    ".pptx": PPTXLoader(),
}

class DocumentParser:

    @staticmethod
    def parse(file_path: str) -> str:

        try:
            # Detect extension
            extension = (
                Path(file_path)
                .suffix
                .lower()
            )

            # Check custom loader (.txt, .csv, .json, .xml)
            loader = CUSTOM_LOADERS.get(
                extension
            )

            if loader:

                print(
                    f"[CUSTOM LOADER] {extension} -> {file_path}"
                )

                return loader.load(
                    file_path
                )

            # Fallback to Unstructured (.pdf, .docx, .pptx, .xlsx, .html, .eml/.msg, .md)
            print(
                f"[UNSTRUCTURED] {extension} -> {file_path}"
            )

            elements = partition(
                filename=file_path
            )

            # Merge extracted elements
            text = "\n".join(
                str(element)
                for element in elements
                if str(element).strip()
            )

            return text

        except Exception as e:

            print(
                f"[PARSER ERROR] {file_path}"
            )

            print(str(e))

            return ""