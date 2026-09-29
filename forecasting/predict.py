import pandas as pd

def predict(df: pd.DataFrame) -> pd.DataFrame:
    """In: columns phc_id, medicine, date. Out: same + forecast_units."""
    out = df.copy()
    out["forecast_units"] = 100  # placeholder; LightGBM goes here later
    return out