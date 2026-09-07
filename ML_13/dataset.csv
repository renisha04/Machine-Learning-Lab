%%writefile app.py

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

class DummyModel:
    def predict(self, X):
        return ["Class A"]

model = DummyModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
