import pandas as pd

from ingestion.loaders.base_loader import BaseLoader


class CSVLoader(BaseLoader):

    def load(self, file_path):

        df = pd.read_csv(
            file_path,
            encoding_errors="ignore"
        )

        return df.to_string()