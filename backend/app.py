from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_pipeline import ask_rag

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "RAG Backend Running 🚀  ( Backend is connected successfully)"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    message = data.get("message", "")

    try:
        reply = ask_rag(message)

    except Exception as e:
        reply = f"ERROR: {str(e)}"

    return jsonify({
        "reply": reply
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)