from dataclasses import dataclass


@dataclass
class ParsedDocument:

    text: str

    page_number: int = -1

    sheet_name: str = ""

    slide_number: int = -1

    section_name: str = ""