from ingestion.loaders.base_loader import (
    BaseLoader
)


class TXTLoader(BaseLoader):

    def load(self, file_path):

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return f.read()