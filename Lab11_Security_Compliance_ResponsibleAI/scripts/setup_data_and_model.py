from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    group = rng.choice(["Group_A", "Group_B"], size=n_samples, p=[0.6, 0.4])
    income = rng.normal(loc=60000, scale=15000, size=n_samples).clip(20000, 150000)
    credit_score = rng.normal(loc=680, scale=60, size=n_samples).clip(300, 850)
    debt_to_income = rng.uniform(0.05, 0.65, size=n_samples)

    # Underlying qualification score with slight historical disparity on Group_B
    group_penalty = np.where(group == "Group_B", -0.65, 0.0)
    logits = (
        0.00003 * (income - 55000)
        + 0.015 * (credit_score - 660)
        - 2.2 * (debt_to_income - 0.30)
        + group_penalty
        + rng.normal(0, 0.5, size=n_samples)
    )
    probs = 1 / (1 + np.exp(-logits))
    approved = (probs >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "group": group,
            "income": np.round(income, 2),
            "credit_score": np.round(credit_score, 0),
            "debt_to_income": np.round(debt_to_income, 3),
            "approved": approved,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data()
    train_df, val_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["approved"]
    )

    feature_cols = ["income", "credit_score", "debt_to_income"]
    X_train, y_train = train_df[feature_cols], train_df["approved"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    val_path = DATA_DIR / "validation_data.csv"
    model_path = MODELS_DIR / "dummy_model.joblib"

    val_df.to_csv(val_path, index=False)
    joblib.dump(model, model_path)

    print(f"Saved validation dataset ({len(val_df)} rows) -> {val_path}")
    print(f"Saved trained model -> {model_path}")


if __name__ == "__main__":
    main()