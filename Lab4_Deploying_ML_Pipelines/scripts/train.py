import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml.data import load_data  # type: ignore # pylint: disable=wrong-import-position,import-error # noqa: E402
from src.ml.train import save_model, train_model  # type: ignore # pylint: disable=wrong-import-position,import-error # noqa: E402


def main(config_path: str):
    """Main training script orchestrated by MLflow."""
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ml-lab4-training")

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with config_path.open() as f:
        config = yaml.safe_load(f)

    # Start an MLflow run
    with mlflow.start_run():
        print("Starting MLflow run...")

        # Log parameters
        mlflow.log_params(config["parameters"])
        print(f"Logged parameters: {config['parameters']}")

        # Load data
        data_path = (project_root / config["data"]["path"]).resolve()
        df = load_data(str(data_path))
        print(f"Data loaded from {data_path}")

        # Train model
        model = train_model(df)
        print("Model trained successfully.")

        # Log metrics (dummy metrics for demonstration)
        metrics = {"mse": 0.1, "r2_score": 0.95}
        mlflow.log_metrics(metrics)
        print(f"Logged metrics: {metrics}")

        # Save and log the model artifact
        model_output_path = (project_root / config["model"]["path"]).resolve() / "model.pkl"
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(model_output_path))
        print(f"Model saved to {model_output_path}")

        # Log the model with MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model logged to MLflow.")

        print("MLflow run completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file.")
    args = parser.parse_args()
    main(args.config)
