import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model(df: pd.DataFrame):
    """Trains a simple Linear Regression model."""
    X = df[["feature_1", "feature_2"]]
    y = df["target"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def save_model(model, path: str):
    """Saves the trained model to a file."""
    joblib.dump(model, path)
