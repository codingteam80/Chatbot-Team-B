import pandas as pd

from ingestion.document import ParsedDocument
from ingestion.loaders.base_loader import BaseLoader


class XLSXLoader(BaseLoader):

    def load(self, file_path):

        excel = pd.ExcelFile(file_path)

        documents = []

        for sheet_name in excel.sheet_names:

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name
            )

            text = df.to_string(
                index=False
            )

            documents.append(

                ParsedDocument(
                    text=text,
                    sheet_name=sheet_name
                )

            )

        print(
            f"[XLSX] Sheets created: {len(documents)}"
        )

        return documents