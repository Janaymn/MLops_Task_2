import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import pickle

# Load training data
train_df = pd.read_csv('data/train1.csv')
print("Train data loaded successfully, shape:", train_df.shape)

# Separate features and target
X = train_df.drop(columns=['species'])
y = train_df['species']

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)
print("Model trained successfully")

# Save trained model
os.makedirs('models', exist_ok=True)
with open('models/model1.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully to models/model1.pkl")
