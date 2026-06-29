import xml.etree.ElementTree as ET

from ingestion.loaders.base_loader import BaseLoader


class XMLLoader(BaseLoader):

    def load(self, file_path):

        tree = ET.parse(file_path)

        root = tree.getroot()

        return ET.tostring(
            root,
            encoding="unicode"
        )