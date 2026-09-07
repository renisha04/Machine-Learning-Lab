import requests


BASE_URL = "http://127.0.0.1:5000"


# ============================================================
# TEST HOME API
# ============================================================

print("\n======================================")
print("TESTING HOME API")
print("======================================")

response = requests.get(
    BASE_URL
)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())


# ============================================================
# GET FEATURES
# ============================================================

print("\n======================================")
print("GETTING MODEL FEATURES")
print("======================================")

response = requests.get(
    BASE_URL + "/features"
)

print("Status Code:", response.status_code)

feature_data = response.json()

print(
    "Number of features:",
    feature_data["number_of_features"]
)


# ============================================================
# CREATE SAMPLE INPUT
# ============================================================

number_of_features = feature_data[
    "number_of_features"
]


# Use average-ish values as a simple test input.
sample_features = [
    14.0,
    20.0,
    90.0,
    600.0,
    0.10,
    0.10,
    0.08,
    0.05,
    0.18,
    0.06,
    0.4,
    1.0,
    3.0,
    40.0,
    0.006,
    0.02,
    0.02,
    0.01,
    0.02,
    0.003,
    16.0,
    25.0,
    105.0,
    800.0,
    0.13,
    0.25,
    0.25,
    0.12,
    0.30,
    0.08
]


# Make sure number of values matches model
if len(sample_features) != number_of_features:

    print("\nSample input has wrong number of values.")

    # Automatically create a valid input
    sample_features = [
        0.0
        for _ in range(number_of_features)
    ]


# ============================================================
# TEST PREDICTION API
# ============================================================

print("\n======================================")
print("TESTING PREDICTION API")
print("======================================")


payload = {
    "features": sample_features
}


response = requests.post(
    BASE_URL + "/predict",
    json=payload
)


print("Status Code:", response.status_code)

print("\nPrediction Result:")

print(response.json())


print("\n======================================")
print("API TEST COMPLETED")
print("======================================")
