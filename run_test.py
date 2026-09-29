import pandas as pd
from forecasting.predict import predict
from optimization.optimize import optimize

df = pd.DataFrame({
    "phc_id": [1, 2],
    "medicine": ["paracetamol", "ors"],
    "date": ["2026-10-01", "2026-10-01"],
})
plan = optimize(predict(df), {"max_per_phc": 80})
print(plan)