from flask import Flask, request, jsonify
import pickle
import numpy as np


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("features.pkl", "rb") as file:
        features = pickle.load(file)

    print("Model loaded successfully!")

except FileNotFoundError:

    print("\nERROR:")
    print("Model files are missing.")
    print("Run this first:")
    print("python trainmodel.py")

    model = None
    scaler = None
    features = []


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Breast Cancer Prediction API",
        "status": "running",
        "model": "Logistic Regression",
        "endpoint": "/predict"
    })


# ============================================================
# FEATURES ROUTE
# ============================================================

@app.route("/features", methods=["GET"])
def get_features():

    return jsonify({
        "number_of_features": len(features),
        "features": features
    })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check model
        if model is None:

            return jsonify({
                "error": "Model not loaded. Run trainmodel.py first."
            }), 500


        # Get JSON
        data = request.get_json()


        if data is None:

            return jsonify({
                "error": "Please send JSON data."
            }), 400


        # Get features
        input_features = data.get("features")


        if input_features is None:

            return jsonify({
                "error": "Missing 'features' field."
            }), 400


        # Check number of features
        if len(input_features) != len(features):

            return jsonify({
                "error": (
                    f"Expected {len(features)} features "
                    f"but received {len(input_features)}."
                )
            }), 400


        # Convert to NumPy array
        input_array = np.array(
            input_features,
            dtype=float
        ).reshape(1, -1)


        # Scale input
        scaled_input = scaler.transform(
            input_array
        )


        # Make prediction
        prediction = model.predict(
            scaled_input
        )[0]


        # Prediction probability
        probabilities = model.predict_proba(
            scaled_input
        )[0]


        # Determine diagnosis
        if prediction == 1:

            diagnosis = "Malignant"

            confidence = probabilities[1] * 100

        else:

            diagnosis = "Benign"

            confidence = probabilities[0] * 100


        # Return result
        return jsonify({

            "status": "success",

            "prediction": int(prediction),

            "diagnosis": diagnosis,

            "confidence": round(
                float(confidence),
                2
            ),

            "message": (
                "The model predicts Malignant."
                if prediction == 1
                else
                "The model predicts Benign."
            )

        })


    except ValueError:

        return jsonify({
            "error": "All feature values must be numeric."
        }), 400


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("BREAST CANCER PREDICTION API")
    print("======================================")

    print("Server: http://127.0.0.1:5000")

    print("======================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
