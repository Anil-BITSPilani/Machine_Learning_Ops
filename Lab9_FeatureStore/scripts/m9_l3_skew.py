import pandas as pd
from m9_l2_online_store import compute_30d_features

def compute_7d_features_bug(transactions_df, customer_id, as_of_time, window_days=7):
    # This is the buggy implementation with a 7-day window
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
        "total_7d_spend_bug": float(total),
        "count_7d_txn_bug": int(count),
        "avg_7d_ticket_bug": float(avg),
    }

def main():
    transactions = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
    offline_features_df = pd.read_csv("offline_features.csv")
    as_of_time = transactions["timestamp"].max()
    example_cid = 1

    # --- Step 1: Show correct offline values ---
    print("--- Step 1: Offline Features (Ground Truth for Training) ---")
    offline_row = offline_features_df[offline_features_df["customer_id"] == example_cid]
    print(offline_row[["total_30d_spend", "count_30d_txn", "avg_30d_ticket"]].to_string(index=False))

    # --- Step 2: Introduce a wrong online implementation ---
    print("\n--- Step 2: Buggy Online Features (7-day window) ---")
    buggy_online_features = compute_7d_features_bug(transactions, example_cid, as_of_time)
    print(buggy_online_features)
    print("\n*** NOTICE: The values are different! This is training-serving skew. ***")

    # --- Step 3: Fix skew by reusing the shared definition ---
    print("\n--- Step 3: Fixed Online Features (Reusing 30-day logic) ---")
    fixed_online_features = compute_30d_features(transactions, example_cid, as_of_time)
    print(fixed_online_features)
    print("\n*** NOTICE: The values now match the offline features. Skew is eliminated. ***")

if __name__ == "__main__":
    main()
