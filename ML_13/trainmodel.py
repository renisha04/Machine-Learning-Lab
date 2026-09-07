import os
import pickle
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# STEP 1: CREATE DATASET.CSV
# ============================================================

dataset_file = "dataset.csv"

print("\n======================================")
print("BREAST CANCER PREDICTION")
print("======================================")

print("\nChecking dataset...")


# Create a correct dataset if dataset.csv does not exist
# or if the existing file is invalid.
create_dataset = True

if os.path.exists(dataset_file):

    try:
        existing_data = pd.read_csv(dataset_file)

        required_columns = ["id", "diagnosis"]

        if all(column in existing_data.columns for column in required_columns):
            if len(existing_data) > 100:
                create_dataset = False
                print("Existing dataset is valid.")

    except Exception:
        print("Existing dataset is invalid.")


if create_dataset:

    print("Creating a new dataset.csv...")

    breast_cancer = load_breast_cancer()

    X_original = breast_cancer.data
    y_original = breast_cancer.target

    feature_names = list(breast_cancer.feature_names)

    # Create DataFrame
    data = pd.DataFrame(
        X_original,
        columns=feature_names
    )

    # Add ID
    data.insert(
        0,
        "id",
        range(1, len(data) + 1)
    )

    # Scikit-learn:
    # 0 = malignant
    # 1 = benign
    #
    # We convert it to:
    # M = malignant
    # B = benign

    data.insert(
        1,
        "diagnosis",
        ["M" if value == 0 else "B" for value in y_original]
    )

    data.to_csv(
        dataset_file,
        index=False
    )

    print("dataset.csv created successfully!")

else:

    data = pd.read_csv(dataset_file)


# ============================================================
# STEP 2: DISPLAY DATASET INFORMATION
# ============================================================

print("\n======================================")
print("DATASET INFORMATION")
print("======================================")

print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# STEP 3: PREPARE DATA
# ============================================================

print("\nPreparing data...")


# Remove unnecessary ID column
if "id" in data.columns:
    data = data.drop("id", axis=1)


# Remove unnecessary column if present
if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)


# Input features
X = data.drop("diagnosis", axis=1)


# Target
y = data["diagnosis"]


# Convert:
# M = 1
# B = 0

y = y.map({
    "M": 1,
    "B": 0
})


# ============================================================
# STEP 4: SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# STEP 5: SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# STEP 7: TEST MODEL
# ============================================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======================================")
print("MODEL RESULTS")
print("======================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)


# ============================================================
# STEP 8: SAVE MODEL
# ============================================================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


with open("features.pkl", "wb") as file:
    pickle.dump(
        list(X.columns),
        file
    )


print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("======================================")

print("\nCreated files:")

print("1. dataset.csv")
print("2. model.pkl")
print("3. scaler.pkl")
print("4. features.pkl")

print("\nYou can now run:")
print("python app.py")
