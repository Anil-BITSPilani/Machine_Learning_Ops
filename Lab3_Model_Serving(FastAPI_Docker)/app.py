from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# --- 1. Load the Model ---
# The model is loaded once when the application starts
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    model = None # Handle case where model is not found

# --- 2. Define the Request Body Schema ---
# This ensures the input data is validated and documented automatically by FastAPI/Pydantic
class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float

# --- 3. Initialize the FastAPI Application ---
app = FastAPI(
    title="ML Model Serving API",
    description="A simple FastAPI service to serve a scikit-learn Linear Regression model.",
    version="1.0.0"
)

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    """
    A simple health check endpoint.
    """
    return {"message": "ML Model Serving API is running. Visit /docs for API documentation."}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint to get a prediction from the loaded ML model.
    Expects a JSON body with 'feature_1' and 'feature_2'.
    """
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    # Convert the Pydantic model data into a format the scikit-learn model expects (a 2D array)
    data_point = np.array([[request.feature_1, request.feature_2]])

    # Make prediction
    prediction = model.predict(data_point)[0]

    # Return the result
    return {
        "input_data": request.model_dump(),
        "prediction": prediction.item()  # .item() converts numpy float to standard Python float
    }

# --- 5. Example of a simple synchronous endpoint for model info ---
@app.get("/model_info")
def get_model_info():
    """
    Returns basic information about the loaded model.
    """
    if model is None:
        return {"error": "Model not loaded."}

    return {
        "model_name": "LinearRegression",
        "version": "1.0.0",
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.item()
    }
