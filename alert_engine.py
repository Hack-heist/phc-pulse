"""
===============================================================================
 PHC-PULSE - STOCK-OUT ALERT ENGINE
===============================================================================

WHAT THIS SCRIPT DOES
---------------------
Turns a demand forecast into an early stock-out warning (Capability 3 of the
project: "generate early warnings for potential stock-outs").

The logic is deliberately simple and transparent, so a health official can
audit it — the intelligence lives in the forecast, not here:

    days_remaining = current_stock / forecasted_daily_demand

    CRITICAL : days_remaining < lead_time_days
               (stock dies BEFORE a reorder could arrive -> reorder now)
    WATCH    : lead_time_days <= days_remaining <= lead_time_days * 1.5
               (getting close, not urgent yet -> plan a reorder)
    OK       : stock comfortably outlasts the lead time
    If forecasted_daily_demand is 0, demand is negligible: stock never
    depletes, so the status is OK (and we avoid dividing by zero).

CONNECTION TO THE REST OF THE PROJECT
-------------------------------------
`forecasted_daily_demand` will come from the trained model:

    from train_forecast_model import predict_demand
    forecast = predict_demand("PHC_01", "antimalarial", recent_30_days)
    check_alert("PHC_01", "antimalarial", current_stock, 7, sum(forecast) / len(forecast))

HOW TO RUN
----------
From the project folder (venv active):
    python alert_engine.py
No third-party libraries needed - pure Python.
===============================================================================
"""

# =============================================================================
# SECTION 1: SETTINGS
# =============================================================================

WATCH_BUFFER = 1.5   # WATCH band = lead_time_days * this multiplier
DAYS_DECIMALS = 1    # decimal places for days_remaining in the output


# =============================================================================
# SECTION 2: THE ALERT FUNCTION
# =============================================================================

def check_alert(phc_id, drug_name, current_stock, lead_time_days, forecasted_daily_demand):
    """
    Decide whether one PHC's stock of one drug is CRITICAL, WATCH or OK.

    Logic: work out how many days the current stock will last at the
    forecasted daily demand, then compare that against the lead time
    (how long a reorder takes to arrive).

      - CRITICAL: stock runs out before the reorder can arrive
                  (days_remaining < lead_time_days)
      - WATCH:    stock gets close to that danger zone
                  (up to WATCH_BUFFER x lead_time_days)
      - OK:       stock comfortably outlasts the lead time
      - forecasted_daily_demand <= 0: demand is negligible, stock will not
        deplete, so we return OK without dividing by zero.

    Args:
        phc_id:                   e.g. "PHC_01" (used in the message only)
        drug_name:                e.g. "antimalarial" (used in the message only)
        current_stock:            units on the shelf right now
        lead_time_days:           days between placing a reorder and it arriving
        forecasted_daily_demand:  predicted units consumed per day (from the model)

    Returns:
        dict with keys:
            status         - "CRITICAL", "WATCH" or "OK"
            days_remaining - stock life in days, rounded to 1 decimal
                             (None when demand is negligible)
            message        - human-readable explanation for a dashboard
    """
    # --- Guard: zero (or negative) forecast avoids division by zero ---
    if forecasted_daily_demand <= 0:
        return {
            "status": "OK",
            "days_remaining": None,
            "message": (f"{drug_name} demand is negligible at {phc_id} "
                        f"({forecasted_daily_demand} units/day) - no stock-out risk."),
        }

    days_remaining = current_stock / forecasted_daily_demand
    days = round(days_remaining, DAYS_DECIMALS)
    watch_limit = lead_time_days * WATCH_BUFFER

    if days_remaining < lead_time_days:
        status = "CRITICAL"
        message = (f"{drug_name} at {phc_id} will last approximately {days} more days; "
                   f"lead time is {lead_time_days} days - reorder now.")
    elif days_remaining <= watch_limit:
        status = "WATCH"
        message = (f"{drug_name} at {phc_id} will last approximately {days} more days; "
                   f"lead time is {lead_time_days} days - getting low, plan a reorder soon.")
    else:
        status = "OK"
        message = (f"{drug_name} at {phc_id} will last approximately {days} more days; "
                   f"lead time is {lead_time_days} days - stock is healthy.")

    return {"status": status, "days_remaining": days, "message": message}


# =============================================================================
# SECTION 3: QUICK SELF-TEST
# The numbers match the data generator's baselines, so the scenarios are
# realistic: antimalarial demand ~10 units/day (low-burden PHC_01 in
# District_A) or ~30 units/day (high-burden PHC_14 in District_D), and
# respiratory demand ~40 units/day (PHC_09 in District_C).
# =============================================================================

if __name__ == "__main__":
    tests = [
        # 32 stock / 10 per day = 3.2 days of life, refill needs 7 -> CRITICAL
        ("CRITICAL", check_alert("PHC_01", "antimalarial", 32, 7, 10)),
        # 261 stock / 30 per day = 8.7 days, inside the 7-10.5 band -> WATCH
        ("WATCH",    check_alert("PHC_14", "antimalarial", 261, 7, 30)),
        # 1920 stock / 40 per day = 48 days, far beyond 10.5 -> OK
        ("OK",       check_alert("PHC_09", "respiratory", 1920, 7, 40)),
    ]

    for expected, result in tests:
        print(f"\n[{expected}] -> status={result['status']}  "
              f"days_remaining={result['days_remaining']}")
        print(f"  {result['message']}")
