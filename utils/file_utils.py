from config.settings import (
    DOCUMENT_DIR,
    SUPPORTED_EXTENSIONS
)


def get_all_documents():

    """
    Return supported documents in a deterministic order.

    Stable ordering makes logs, manifests, and full rebuilds
    easier to compare during debugging.
    """

    files = []

    for file_path in DOCUMENT_DIR.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            files.append(
                file_path
            )

    return sorted(
        files,
        key=lambda path: str(path).lower()
    )
