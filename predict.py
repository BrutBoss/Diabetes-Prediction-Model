import pickle
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "model.bin"

# Load trained model (Pipeline: scaler + model)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask("diabetes-prediction")


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()

    if patient is None:
        return jsonify({"error": "No input data provided"}), 400

    # Convert input JSON to DataFrame
    df = pd.DataFrame([patient])

    # One-hot encode (must match training)
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict probability
    probability = model.predict_proba(df)[0, 1]
    prediction = probability >= 0.5

    result = {
        "diabetes_probability": round(float(probability), 4),
        "diabetes_prediction": bool(prediction)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
