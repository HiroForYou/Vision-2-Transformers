import sys

from flask import Flask, jsonify, request
from flask_cors import CORS
from model.ViT import inference_ViT

app = Flask(__name__)
CORS(app)


@app.route("/ViT", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    outputs = inference_ViT(data)
    return jsonify({"outputs": outputs})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"response": "This is Vision Transfomer Application"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, port=5000)
