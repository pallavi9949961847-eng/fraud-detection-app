import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("creditcard.csv")

# TAKE SMALL SAMPLE (FAST)
data = data.sample(10000)

# Split
X = data.drop("Class", axis=1)
y = data["Class"]

# Train model
model = RandomForestClassifier(n_estimators=10)  # smaller trees
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model created successfully!")