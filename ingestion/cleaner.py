import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:

        # Empty protection
        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", " ")

        # Normalize line endings
        text = re.sub(r"\r", "\n", text)

        # Unicode cleanup
        text = text.replace("\u2018", "'")
        text = text.replace("\u2019", "'")

        text = text.replace("\u201c", '"')
        text = text.replace("\u201d", '"')

        text = text.replace("\u2013", "-")
        text = text.replace("\u2014", "-")

        text = text.replace("\u2022", "-")

        text = text.replace("\xa0", " ")

        # Remove page numbers
        text = re.sub(
            r"(?im)^page\s+\d+\s*$",
            "",
            text
        )

        text = re.sub(
            r"(?im)^page\s+\d+\s+of\s+\d+\s*$",
            "",
            text
        )

        # Remove long separators
        text = re.sub(
            r"[-_=]{5,}",
            "",
            text
        )

        # Reduce excessive new lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        # Remove multiple spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Remove trailing spaces
        text = re.sub(
            r"\s+\n",
            "\n",
            text
        )

        # Remove empty lines with spaces
        text = re.sub(
            r"\n\s+\n",
            "\n\n",
            text
        )

        # Final cleanup
        text = text.strip()

        return text