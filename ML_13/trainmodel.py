import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data.csv")

X = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
y = data["diagnosis"]

y = y.map({"M": 1, "B": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model trained successfully")
print("Accuracy:", accuracy)

with open("model.pkl", "wb") as file:
    pickle.dump((scaler, model), file)

print("Model saved successfully")
