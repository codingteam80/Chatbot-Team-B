from pathlib import Path

from config.settings import (
    DOCUMENT_DIR,
    SUPPORTED_EXTENSIONS
)


def get_all_documents():

    files = []

    for file in DOCUMENT_DIR.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            files.append(file)

    return files