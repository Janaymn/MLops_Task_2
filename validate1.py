import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

print("Script started")

# Load model
with open('models/model1.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully")

# Load test data
test_df = pd.read_csv('data/test1.csv')
print("Test data loaded successfully, shape:", test_df.shape)

# Separate features and target
X = test_df.drop(columns=['species'])
y = test_df['species']

# Predictions
preds = model.predict(X)
print("Predictions done")

# Save metrics
acc = accuracy_score(y, preds)
print("Accuracy calculated:", acc)


with open('metrics1.json', 'w') as f:
    json.dump({'accuracy': acc}, f)
print("Metrics saved")

# Confusion matrix
cm = confusion_matrix(y, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig('confusion_matrix1.png')
print("Confusion matrix saved as confusion_matrix1.png")