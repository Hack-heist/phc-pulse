import pandas as pd

def optimize(forecast_df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """In: forecast + {'max_per_phc': int}. Out: same + ship_units."""
    out = forecast_df.copy()
    out["ship_units"] = out["forecast_units"].clip(upper=constraints["max_per_phc"])
    return out