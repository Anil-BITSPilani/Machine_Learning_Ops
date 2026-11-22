import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Create a dummy dataset
data = {
    'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature_2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'target': [12, 14, 16, 18, 20, 22, 24, 26, 28, 30] # target = 2*f1 + f2
}
df = pd.DataFrame(data)

X = df[['feature_1', 'feature_2']]
y = df['target']

# 2. Train a simple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 3. Save the model
model_path = 'model.pkl'
joblib.dump(model, model_path)

print(f"Model trained and saved to {model_path}")