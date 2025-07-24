import os
import gc
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from run_pipeline import vector_from_feats, bounded_targets
import joblib

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(__file__).resolve().parent / "models"

def load_model(target):
    model_path = MODEL_DIR / f"model_{target}.joblib"
    return joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    track_key = data.get("track", "unknown")
    feats = data.get("features")

    if feats is None:
        return jsonify({"error": "Missing 'features' in request"}), 400

    try:
        targets = bounded_targets.union({
            "danceability", "energy", "acousticness", "valence", "tempo", "loudness",
            "instrumentalness", "speechiness", "liveness", "key"
        })

        predictions = {}
        for target in targets:
            try:
                model = load_model(target)
            except Exception as e:
                logging.warning(f"Could not load model {target}: {e}")
                predictions[target] = None
                continue

            X_raw = vector_from_feats(feats, target)
            if X_raw is None or np.isnan(X_raw).any() or X_raw.shape[1] == 0:
                predictions[target] = None
            else:
                pred = model.predict(X_raw)[0]
                if target == "key":
                    pred = int(round(pred)) % 12
                elif target in bounded_targets:
                    pred = float(np.clip(pred, 0, 1))
                else:
                    pred = float(pred)
                predictions[target] = pred

            del model
            gc.collect()

        return jsonify({"track": track_key, "features": predictions})

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e), "track": track_key}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    app.run(host="0.0.0.0", port=port)
