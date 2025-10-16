# app.py - deploy RandomForest model as Flask API
from flask import Flask, request, jsonify, send_from_directory
import joblib, pandas as pd, os
from pathlib import Path

MODEL_PATH = os.environ.get("MODEL_PATH", "rf_model.pkl")

app = Flask(__name__, static_folder='.', static_url_path='')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("⚠️ Warning: could not load model:", e)

@app.route("/", methods=["GET"])
def home():
    idx = Path(__file__).parent / "index.html"
    if idx.exists():
        return send_from_directory(Path(__file__).parent, "index.html")
    return jsonify({"message": "Index page not found. Place index.html in project root."}), 404

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    if "instances" not in data and "instance" not in data:
        return jsonify({"error": "Send 'instance' or 'instances' in JSON"}), 400

    records = [data["instance"]] if "instance" in data else data["instances"]
    df = pd.DataFrame(records)
    preds = model.predict(df).tolist()
    response = {"predictions": preds}
    if hasattr(model, "predict_proba"):
        response["probabilities"] = model.predict_proba(df).tolist()
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
