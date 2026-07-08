from flask import Flask, request
import os

app = Flask(__name__)

DOCUMENT_FOLDER = r"C:\user_dev\company-chatbot\data\all_documents"


@app.route("/open")
def open_file():

    filename = request.args.get("file")

    if not filename:
        return "Missing filename", 400

    path = os.path.join(
        DOCUMENT_FOLDER,
        filename
    )

    if not os.path.exists(path):
        return "File not found", 404

    os.startfile(path)

    return "OK"


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8502
    )