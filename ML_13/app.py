from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Load feature names
with open("features.pkl", "rb") as file:
    features = pickle.load(file)


@app.route("/")
def home():
    return jsonify({
        "message": "Breast Cancer Prediction API is running!",
        "status": "success"
    })


@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({
        "features": features,
        "number_of_features": len(features)
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON data received"
            }), 400


        # Get input features
        input_features = data.get("features")


        if input_features is None:
            return jsonify({
                "error": "Please provide 'features'"
            }), 400


        # Check number of features
        if len(input_features) != len(features):
            return jsonify({
                "error": f"Expected {len(features)} features, but received {len(input_features)}"
            }), 400


        # Convert input to numpy array
        input_array = np.array(input_features, dtype=float).reshape(1, -1)


        # Scale input
        scaled_input = scaler.transform(input_array)


        # Prediction
        prediction = model.predict(scaled_input)[0]


        # Probability
        probability = model.predict_proba(scaled_input)[0]


        if prediction == 1:
            diagnosis = "Malignant"
        else:
            diagnosis = "Benign"


        confidence = float(max(probability) * 100)


        return jsonify({
            "prediction": int(prediction),
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2)
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("\n================================")
    print("Breast Cancer Prediction API")
    print("================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("================================\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
