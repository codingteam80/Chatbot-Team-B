import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from services.answer_service import (
    AnswerService
)

assistant = AnswerService()

while True:

    question = input(
        "\nQuestion: "
    )

    if question.lower() in [
        "exit",
        "quit"
    ]:
        break

    response = assistant.ask(
        question
    )

    print("\nAnswer:")
    print(
        response["answer"]
    )

    print("\nSources:")

    for source in response[
        "sources"
    ]:

        print(
            f"- {source}"
        )