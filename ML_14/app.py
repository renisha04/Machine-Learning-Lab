from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = data['features']

    if len(features) != 30:
        return jsonify({"error": "Expected 30 features"}), 400

    features = np.array(features, dtype=float).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": prediction
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
