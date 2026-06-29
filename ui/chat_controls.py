from __future__ import annotations

import json
from datetime import datetime


class ChatControls:
    """
    ==========================================================
    ChatControls

    Utility functions for chat actions.

    This class does NOT render any UI.

    Responsibilities:
        • Export conversation as TXT
        • Export conversation as JSON
        • Generate filenames

    UI buttons should be created in app.py.

    ==========================================================
    """

    # ======================================================
    # TIMESTAMP
    # ======================================================

    @staticmethod
    def timestamp():

        return datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    # ======================================================
    # TXT EXPORT
    # ======================================================

    @staticmethod
    def export_txt(messages):

        lines = []

        for message in messages:

            role = message.get(
                "role",
                "assistant"
            ).upper()

            content = message.get(
                "content",
                ""
            )

            lines.append(
                f"{role}\n{content}\n"
            )

        return "\n".join(lines)

    # ======================================================
    # JSON EXPORT
    # ======================================================

    @staticmethod
    def export_json(messages):

        return json.dumps(

            messages,

            indent=4,

            ensure_ascii=False

        )

    # ======================================================
    # TXT FILE NAME
    # ======================================================

    @staticmethod
    def txt_filename():

        return (
            f"conversation_"
            f"{ChatControls.timestamp()}.txt"
        )

    # ======================================================
    # JSON FILE NAME
    # ======================================================

    @staticmethod
    def json_filename():

        return (
            f"conversation_"
            f"{ChatControls.timestamp()}.json"
        )