from pathlib import Path
import logging

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("DocuBot")

if not logger.handlers:

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # File
    file = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)


def log(message):

    logger.info(message)

def separator(title=""):

    line = "=" * 60

    logger.info("")
    logger.info(line)

    if title:
        logger.info(title)

    logger.info(line)


def summary(**kwargs):

    logger.info("")

    logger.info("=" * 60)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 60)

    for key, value in kwargs.items():

        logger.info(
            f"{key:<18}: {value}"
        )

    logger.info("=" * 60)