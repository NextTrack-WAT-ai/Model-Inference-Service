from flask import Flask, request, jsonify
import logging
import os
from joblib import load
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(__file__).resolve().parent / "models"
loaded_models = {}

def load_model(target):
    if target not in loaded_models:
        model_path = MODEL_DIR / f"model_{target}.joblib"
        try:
            loaded_models[target] = load(model_path)
            logging.info(f"Loaded model: {target}")
        except FileNotFoundError:
            logging.warning(f"Model not found: {model_path}")
            return None
    return loaded_models[target]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing features"}), 400

    predictions = {}
    for target in [
        "danceability", "energy", "acousticness", "valence",
        "loudness", "speechiness", "instrumentalness", "liveness",
        "tempo", "key"
    ]:
        model = load_model(target)
        if model is None:
            continue
        try:
            value = model.predict([features])[0]
            predictions[target] = float(value)
        except Exception as e:
            logging.warning(f"Prediction failed for {target}: {e}")
            continue

    return jsonify(predictions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
