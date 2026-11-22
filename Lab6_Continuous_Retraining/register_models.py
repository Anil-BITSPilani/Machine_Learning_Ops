"""
Register existing MLflow runs to the Model Registry
"""
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")

def register_existing_runs():
    """Find existing runs and register them to the model registry"""
    
    client = mlflow.tracking.MlflowClient()
    model_name = "credit_risk_model"
    
    # Get all runs from experiment 1 (default experiment)
    experiment_id = "1"
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    if not runs:
        print(f"No runs found in experiment {experiment_id}")
        return
    
    print(f"Found {len(runs)} runs. Registering them to model '{model_name}'...")
    
    for idx, run in enumerate(runs, 1):
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName", f"Run {idx}")
        
        print(f"\n--- Registering Run {idx} ---")
        print(f"Run ID: {run_id}")
        print(f"Run Name: {run_name}")
        
        # Register the model
        model_uri = f"runs:/{run_id}/model"
        
        try:
            # Register model version
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"✅ Registered as version {mv.version}")
            
            # Promote first version to Production
            if idx == 1:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production"
                )
                print(f"🎉 Promoted version {mv.version} to Production stage")
            else:
                # Keep others in Staging
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Staging"
                )
                print(f"📋 Moved version {mv.version} to Staging stage")
                
        except Exception as e:
            print(f"❌ Error registering model: {e}")
    
    print(f"\n{'='*60}")
    print("Registration complete!")
    print(f"{'='*60}")
    
    # Show final status
    print(f"\nModel Registry Status for '{model_name}':")
    try:
        for stage in ["Production", "Staging", "None"]:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                for v in versions:
                    print(f"  Stage '{stage}': Version {v.version} (Run: {v.run_id[:8]}...)")
    except:
        # If using newer MLflow, stages might be deprecated
        all_versions = client.search_model_versions(f"name='{model_name}'")
        for v in all_versions:
            print(f"  Version {v.version}: Stage '{v.current_stage}' (Run: {v.run_id[:8]}...)")

if __name__ == "__main__":
    print("="*60)
    print("MLflow Model Registration Script")
    print("="*60)
    register_existing_runs()
