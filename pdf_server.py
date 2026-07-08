from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path


def start_pdf_server():

    document_folder = Path("data/all_documents").resolve()

    handler = partial(
        SimpleHTTPRequestHandler,
        directory=str(document_folder)
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", 8502),
        handler
    )

    print("PDF Server running:")
    print("http://127.0.0.1:8502")

    server.serve_forever()