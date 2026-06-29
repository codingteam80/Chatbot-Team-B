from datetime import datetime


def log(message: str):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{timestamp}] {message}"
    )