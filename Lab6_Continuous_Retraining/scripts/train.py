import yaml
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_and_log(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    mlflow.set_experiment(config["run"]["experiment_name"])
    
    with mlflow.start_run(run_name=config["run"]["run_name"]) as run:
        print(f"Starting run: {run.info.run_name}")
        
        # Load data
        df = pd.read_csv(config["data"]["path"])
        X = df[["feature_1", "feature_2"]]
        y = df["target"]
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Log params and metrics (dummy)
        mlflow.log_params(config["model"])
        mlflow.log_metric("mse", 0.1)
        mlflow.log_artifact(config_path)
        
        # Log model and register
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, "credit_risk_model")
        
        print(f"Model trained and registered as version: {registered_model.version}")
        return registered_model.version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_and_log(args.config)