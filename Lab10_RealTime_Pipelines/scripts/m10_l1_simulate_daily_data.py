import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # --- Create a base dataset ---
    rng = np.random.default_rng(seed=42)
    num_days = 5
    rows_per_day = 10
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    
    records = []
    for day_idx, day in enumerate(dates, start=1):
        for i in range(rows_per_day):
            records.append({
                "day": day_idx,
                "timestamp": day + pd.Timedelta(minutes=int(rng.integers(0, 24*60))),
                "customer_id": int(rng.integers(1, 6)),
                "amount": float(rng.integers(10, 200)),
                "label": int(rng.integers(0, 2)),
            })
            
    base_df = pd.DataFrame(records)
    print("--- Base Synthetic Dataset ---")
    print(base_df.head())

    # --- Split into per-day CSVs ---
    data_dir = Path("../data_m10")
    data_dir.mkdir(exist_ok=True)
    
    print("\n--- Splitting into Daily CSV Files ---")
    for day_idx, group in base_df.groupby("day"):
        file_path = data_dir / f"day_{day_idx}.csv"
        group.to_csv(file_path, index=False)
        print(f"Saved {file_path}, rows={len(group)}")
        
    print(f"\nDaily data files created in: {data_dir.resolve()}")

if __name__ == "__main__":
    main()
