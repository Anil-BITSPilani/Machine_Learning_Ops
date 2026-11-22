from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import os
import pandas as pd

# ---- Settings ----
# Set MLflow tracking URI - use default (SQLite database) if not set
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    # Use default tracking URI (SQLite database in current directory)
    pass

# In a real app, use Pydantic's BaseSettings to load from env vars
MODEL_NAME = os.getenv("MODEL_NAME", "credit_risk_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Global variables to store model info
model = None
model_version = None

# ---- Model Loading ----
def load_model_from_registry(name, stage):
    global model_version
    try:
        # Get model version info from registry
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(name, stages=[stage])

        if not model_versions:
            print(f"ERROR: No model found for {name} in stage {stage}")
            return None

        model_version_info = model_versions[0]
        model_version = model_version_info.version
        run_id = model_version_info.run_id

        model_uri = f"models:/{name}/{stage}"
        print(f"\n{'='*60}")
        print(f" Loading Model from Registry")
        print(f"{'='*60}")
        print(f" Model Name: {name}")
        print(f" Model Stage: {stage}")
        print(f" Model Version: {model_version}")
        print(f"  Run ID: {run_id}")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"{'='*60}\n")

        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Successfully loaded model version {model_version} from {stage} stage\n")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

    model = load_model_from_registry(MODEL_NAME, MODEL_STAGE)

    #FastAPI app
app = FastAPI(title="Config Driven Model Serving")

class process_request(BaseModel):
    feature_1: float
    feature_2: float
