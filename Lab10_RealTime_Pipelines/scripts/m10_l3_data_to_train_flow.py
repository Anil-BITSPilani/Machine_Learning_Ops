import json
import pandas as pd
from pathlib import Path
import time
import os
from m10_l2_incremental_ingestion import ingest_new_days, initialize_state, SCRIPT_DIR, ARTIFACTS_DIR, STATE_PATH, TRAINING_PATH

def should_retrain(state, min_new_days: int = 1):
    """Determines if retraining should be triggered based on new data."""
    last_ingested_day = state.get("last_ingested_day", 0)
    last_retrain_day = state.get("last_retrain_day", 0)
    days_since_retrain = last_ingested_day - last_retrain_day
    
    print(f"[DECISION] Days since last retrain: {days_since_retrain}. Minimum required: {min_new_days}.")
    
    if days_since_retrain >= min_new_days:
        print("[DECISION] Retraining condition met.")
        return True
    else:
        print("[DECISION] Not enough new data to trigger retraining.")
        return False

def run_training_pipeline(state):
    """A dummy function to simulate a model training pipeline."""
    print("\n[TRAIN] >>> Starting training pipeline...")
    print("[TRAIN] Loading training data from:", TRAINING_PATH)
    # In a real pipeline, you would load data, train a model, evaluate, and register it.
    time.sleep(2) # Simulate work
    
    new_version = state.get("model_version", 0) + 1
    state["model_version"] = new_version
    state["last_retrain_day"] = state.get("last_ingested_day", 0)
    
    print(f"[TRAIN] <<< Finished training. New model version: {new_version}")
    print(f"[TRAIN] Updated state: last_retrain_day is now {state['last_retrain_day']}")
    return state

def daily_orchestration_job():
    """Orchestrates the daily data-to-train flow."""
    print("=================================================")
    print("====== Running Daily Orchestration Job ======")
    print("=================================================")
    
    # Step 1: Ingest any new data
    ingest_new_days()
    
    # Step 2: Decide whether to retrain
    print("\n--- Checking Retraining Policy ---")
    state = json.loads(STATE_PATH.read_text())
    
    if should_retrain(state):
        # Step 3: If yes, run the training pipeline
        state = run_training_pipeline(state)
        # Save the final state after training
        STATE_PATH.write_text(json.dumps(state, indent=2))
        print("\nFinal state saved after retraining.")
    else:
        print("\nNo retraining was triggered.")
        
    print("\n====== Daily Job Finished ======")

if __name__ == "__main__":
    # Clean up previous artifacts for a fresh run
    if TRAINING_PATH.exists():
        os.remove(TRAINING_PATH)
    if STATE_PATH.exists():
        os.remove(STATE_PATH)
    
    # Simulate the first run of the daily job
    daily_orchestration_job()
    
    # Simulate a second run where no new data is available
    print("\n\n--- Simulating a second run on the same day (no new data) ---")
    daily_orchestration_job()
