import json

from ingestion.loaders.base_loader import BaseLoader


class JSONLoader(BaseLoader):

    def load(self, file_path):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )