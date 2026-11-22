import pandas as pd

def load_data(path: str):
    """Loads training data from a CSV file."""
    df = pd.read_csv(path)
    return df
