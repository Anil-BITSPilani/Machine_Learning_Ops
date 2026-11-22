import pandas as pd
import numpy as np

def main():
    # --- Data Setup ---
    data = {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        "timestamp": pd.to_datetime([
            "2024-01-01", "2024-01-05", "2024-01-20",
            "2024-01-03", "2024-01-25",
            "2024-01-02", "2024-01-10", "2024-01-18", "2024-01-29"
        ]),
        "amount": [50, 20, 100, 30, 70, 10, 20, 40, 80],
        "category": ["A", "B", "A", "A", "B", "A", "B", "B", "A"],
    }
    transactions = pd.DataFrame(data)
    print("--- Raw Transactions ---")
    print(transactions)

    # --- Feature Table Build ---
    as_of_date = transactions["timestamp"].max()
    window_days = 30
    cutoff = as_of_date - pd.Timedelta(days=window_days)
    
    tx_30d = transactions[transactions["timestamp"] > cutoff]
    
    offline_features = (
        tx_30d
        .groupby("customer_id")
        .agg(
            total_30d_spend=("amount", "sum"),
            count_30d_txn=("amount", "count"),
        )
        .reset_index()
    )
    
    offline_features["avg_30d_ticket"] = (
        offline_features["total_30d_spend"] / offline_features["count_30d_txn"]
    )
    
    offline_features["ref_date"] = as_of_date
    
    print("\n--- Offline Feature Table ---")
    print(offline_features)
    
    # Save for next lab
    offline_features.to_csv("offline_features.csv", index=False)
    transactions.to_csv("transactions.csv", index=False)
    print("\nOffline features and transactions saved to CSVs.")

if __name__ == "__main__":
    main()
