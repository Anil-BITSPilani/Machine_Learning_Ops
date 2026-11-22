import pandas as pd

def compute_30d_features(transactions_df, customer_id, as_of_time, window_days=30):
    cutoff = as_of_time - pd.Timedelta(days=window_days)
    
    tx = transactions_df[
        (transactions_df["customer_id"] == customer_id)
        & (transactions_df["timestamp"] > cutoff)
        & (transactions_df["timestamp"] <= as_of_time)
    ]
    
    total = tx["amount"].sum()
    count = tx["amount"].count()
    
    avg = total / count if count > 0 else 0.0
    
    return {
        "customer_id": int(customer_id),
        "as_of_time": str(as_of_time.date()),
        "total_30d_spend": float(total),
        "count_30d_txn": int(count),
        "avg_30d_ticket": float(avg),
    }

def main():
    transactions = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
    as_of_time = transactions["timestamp"].max()
    customer_ids = transactions["customer_id"].unique()
    
    # --- Build the Online Store (Dict) ---
    online_store = {}
    for cid in customer_ids:
        feats = compute_30d_features(transactions, cid, as_of_time)
        online_store[int(cid)] = feats
        
    print("--- Simulated Online Feature Store (Dictionary) ---")
    import json
    print(json.dumps(online_store, indent=2))

    # --- Simulate Online Requests ---
    def get_online_features(customer_id):
        return online_store.get(customer_id, None)

    print("\n--- Simulating Online Feature Lookups ---")
    print("Customer 1:", get_online_features(1))
    print("Customer 2:", get_online_features(2))
    print("Customer 999 (Missing):", get_online_features(999))

if __name__ == "__main__":
    main()
