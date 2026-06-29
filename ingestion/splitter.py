from llama_index.core.node_parser import SentenceSplitter

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class DocumentSplitter:

    def __init__(self):

        # LlamaIndex chunking engine
        self.splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def split(self, text: str):

        # Convert document into chunks
        return self.splitter.split_text(text)