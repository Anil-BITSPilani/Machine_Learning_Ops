import json
from pathlib import Path
import pandas as pd
import os

# Define paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACTS_DIR = SCRIPT_DIR.parent / "artifacts_m10"
DATA_DIR = SCRIPT_DIR.parent / "data_m10"
STATE_PATH = ARTIFACTS_DIR / "ingestion_state.json"
TRAINING_PATH = ARTIFACTS_DIR / "training_data.csv"

def setup_directories():
    """Ensure data and artifacts directories exist."""
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)

def initialize_state():
    """Create the initial state file if it doesn't exist."""
    if not STATE_PATH.exists():
        print("State file not found. Initializing...")
        state = {"last_ingested_day": 0}
        STATE_PATH.write_text(json.dumps(state, indent=2))
        return state
    return json.loads(STATE_PATH.read_text())

def day_from_path(p: Path) -> int:
    """Extracts the day index from a filename like 'day_1.csv'."""
    return int(p.stem.split("_")[1])

def ingest_new_days():
    """Finds new daily data files, appends them to the master training set, and updates the state."""
    print("--- Starting Incremental Ingestion Job ---")
    state = initialize_state()
    last_ingested = state.get("last_ingested_day", 0)
    print(f"Current state: last_ingested_day = {last_ingested}")

    # Find new files to ingest
    day_files = sorted(DATA_DIR.glob("day_*.csv"))
    new_files = [p for p in day_files if day_from_path(p) > last_ingested]

    if not new_files:
        print("No new day files to ingest.")
        return

    print(f"Found {len(new_files)} new files to ingest: {[f.name for f in new_files]}")

    # Load existing training data
    if TRAINING_PATH.exists():
        training_df = pd.read_csv(TRAINING_PATH)
        print(f"Loaded existing training data with {len(training_df)} rows.")
    else:
        training_df = pd.DataFrame()
        print("No existing training data found. Creating new file.")

    # Process and append new files
    max_day_ingested = last_ingested
    for f in new_files:
        day_idx = day_from_path(f)
        df = pd.read_csv(f)
        
        print(f"\nIngesting {f.name}... (Rows: {len(df)})")
        # Simple quality check
        if df.isna().sum().sum() > 0:
            print("WARNING: Null values detected!")
            print(df.isna().sum())
        
        training_df = pd.concat([training_df, df], ignore_index=True)
        max_day_ingested = max(max_day_ingested, day_idx)

    # Save updated training data
    training_df.to_csv(TRAINING_PATH, index=False)
    print(f"\nUpdated training dataset saved. Total rows: {len(training_df)}")

    # Update state
    state["last_ingested_day"] = max_day_ingested
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"Updated state: last_ingested_day = {max_day_ingested}")
    print("--- Ingestion Job Finished ---")

if __name__ == "__main__":
    setup_directories()
    # Run ingestion
    ingest_new_days()
    
    # Run it a second time to show idempotency
    print("\n--- Running ingestion job a second time to demonstrate idempotency ---")
    ingest_new_days()
