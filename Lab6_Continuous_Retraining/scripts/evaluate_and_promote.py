import yaml
import argparse
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X, y, config):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    
    # Simple business metric
    profit = (
        ((y < config["threshold"]) & (preds < config["threshold"])).sum() * config["profit_correct_non_default"]
        - ((y >= config["threshold"]) & (preds < config["threshold"])).sum() * config["loss_missed_default"]
    )
    
    return {"mse": mse, "profit": profit}

def main(config_path: str, challenger_version: int):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    client = mlflow.tracking.MlflowClient()
    model_name = config["champion_challenger"]["model_name"]

    # Load data
    eval_df = pd.read_csv(config["eval_data"]["path"])
    X_eval = eval_df[["feature_1", "feature_2"]]
    y_eval = eval_df["target"]

    # Load models
    champion_stage = config["champion_challenger"]["champion_stage"]
    champion_uri = f"models:/{model_name}/{champion_stage}"
    challenger_uri = f"models:/{model_name}/{challenger_version}"
    champion = mlflow.pyfunc.load_model(champion_uri)
    challenger = mlflow.pyfunc.load_model(challenger_uri)

    # Evaluate
    champion_metrics = evaluate_model(champion, X_eval, y_eval, config["business_logic"])
    challenger_metrics = evaluate_model(challenger, X_eval, y_eval, config["business_logic"])

    print("--- Evaluation Results ---")
    print(f"Champion ({champion_uri}): {champion_metrics}")
    print(f"Challenger (Version {challenger_version}): {challenger_metrics}")

    # Promotion Logic
    # Promote if challenger is better or equal on MSE (with floating point tolerance)
    # AND profit is not worse
    mse_improved = challenger_metrics["mse"] <= champion_metrics["mse"] * 1.001  # 0.1% tolerance
    profit_not_worse = challenger_metrics["profit"] >= champion_metrics["profit"]
    
    if mse_improved and profit_not_worse:
        print("\n✅ Challenger meets promotion criteria. Promoting to Production.")
        print(f"   MSE: {champion_metrics['mse']:.6f} → {challenger_metrics['mse']:.6f}")
        print(f"   Profit: {champion_metrics['profit']} → {challenger_metrics['profit']}")
        
        # Archive old champion
        for mv in client.get_latest_versions(model_name, stages=["Production"]):
            client.transition_model_version_stage(name=model_name, version=mv.version, stage="Archived")
        # Promote new challenger
        client.transition_model_version_stage(name=model_name, version=challenger_version, stage="Production")
        print(f"\n🎉 Version {challenger_version} is now in Production.")
    else:
        print("\n❌ Challenger does not meet promotion criteria. No promotion.")
        print(f"   Champion MSE: {champion_metrics['mse']:.6f} vs Challenger: {challenger_metrics['mse']:.6f}")
        print(f"   Champion Profit: {champion_metrics['profit']} vs Challenger: {challenger_metrics['profit']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--challenger_version", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.challenger_version)
