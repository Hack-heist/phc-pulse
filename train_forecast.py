"""
===============================================================================
 PHC DEMAND FORECASTING - TRAIN A LIGHTGBM MODEL
===============================================================================

WHAT THIS SCRIPT DOES
---------------------
1. Loads phc_consumption.csv (made by data_generator.py).
2. Builds "features" - clues the model uses to guess tomorrow's demand:
   yesterday's demand, demand a week ago, recent averages, and the season.
3. Splits the data by TIME: the model learns from the first 80% of the dates
   and is tested on the final 20% (which it has never seen).
4. Trains a LightGBM model and reports MAPE and RMSE (two error measures).
5. Saves the model to forecast_model.pkl.
6. Provides predict_demand(), which gives a 7-day forecast for one PHC and one
   drug. You can import it from other scripts:

       from train_forecast_model import predict_demand

HOW TO RUN
----------
Install the extra libraries once (inside your venv):
    pip install lightgbm scikit-learn
Then run from the folder that contains phc_consumption.csv:
    python train_forecast_model.py

MAC NOTE: if you see an error mentioning "libomp", run: brew install libomp
===============================================================================
"""

import pickle                      # saves/loads Python objects to/from a file
import numpy as np                 # number crunching
import pandas as pd                # tables (DataFrames)
from lightgbm import LGBMRegressor # the machine-learning model


# =============================================================================
# SECTION 1: SETTINGS
# =============================================================================

CSV_PATH = "phc_consumption.csv"     # input data
MODEL_PATH = "forecast_model.pkl"    # where the trained model is saved

TEST_FRACTION = 0.20                 # last 20% of dates are held out for testing
FORECAST_DAYS = 7                    # how far ahead predict_demand() looks
MIN_HISTORY_DAYS = 14                # need 14 past days to compute the 14-day average
MAPE_WARNING_THRESHOLD = 30.0        # MAPE above this (in %) triggers a warning

MONSOON_MONTHS = [6, 7, 8, 9]        # June-September
WINTER_MONTHS = [11, 12, 1, 2]       # November-February

# Each PHC + drug pair is its own "time series", so all lag/rolling features
# are calculated separately inside each pair (never mixing different PHCs).
GROUP_COLS = ["phc_id", "drug_name"]

# The exact list of columns the model is trained on. The ORDER matters, and
# predict_demand() must use the same list.
FEATURE_COLUMNS = [
    "lag_1", "lag_7", "roll_7", "roll_14",        # past-demand features
    "month", "is_monsoon", "is_winter",           # calendar features
    "district_code", "drug_code",                 # label-encoded categories
]
CATEGORICAL_FEATURES = ["district_code", "drug_code"]  # tell LightGBM these are categories, not quantities


# =============================================================================
# SECTION 2: FEATURE-BUILDING FUNCTIONS
# These are used BOTH for training and inside predict_demand(), so the model
# always sees features built in exactly the same way.
# =============================================================================

def build_features(df):
    """
    Takes a table with columns date, phc_id, drug_name, units_consumed and
    adds the lag, rolling-average and calendar features.
    """
    df = df.copy()                                   # don't modify the caller's table
    df["date"] = pd.to_datetime(df["date"])          # make sure dates are real dates, not text
    # Sort so each PHC/drug's rows are together and in date order. Lags and
    # rolling averages only make sense when rows are in time order.
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)

    # A "group" is one PHC + one drug. Everything below happens inside groups.
    units_by_group = df.groupby(GROUP_COLS)["units_consumed"]

    # --- Lag features: demand from N days ago ---
    df["lag_1"] = units_by_group.shift(1)   # yesterday's demand
    df["lag_7"] = units_by_group.shift(7)   # demand exactly one week ago

    # --- Rolling averages: the average of the previous N days ---
    # IMPORTANT: we shift by 1 BEFORE averaging. Without that, today's own
    # value would sneak into "the average" and the model would effectively be
    # shown the answer (this is called data leakage). With the shift, the
    # 7-day average for a given day uses only the 7 days BEFORE it.
    df["roll_7"] = units_by_group.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_14"] = units_by_group.transform(lambda s: s.shift(1).rolling(14).mean())

    # --- Calendar features ---
    df["month"] = df["date"].dt.month                                   # 1 to 12
    # Flags are stored as 1 (True) / 0 (False), which LightGBM reads easily.
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS).astype(int)
    df["is_winter"] = df["month"].isin(WINTER_MONTHS).astype(int)

    return df


def encode_categories(df, district_map, drug_map):
    """
    Turns district and drug names (text) into whole numbers, because the model
    needs numbers. This is called "label encoding", e.g. District_A -> 0.
    The maps are created once during training and saved with the model, so
    the same name always gets the same number later.
    """
    df = df.copy()
    df["district_code"] = df["district"].map(district_map)
    df["drug_code"] = df["drug_name"].map(drug_map)
    if df["district_code"].isna().any() or df["drug_code"].isna().any():
        raise ValueError("Found a district or drug name that the model was not trained on.")
    return df


# =============================================================================
# SECTION 3: ERROR MEASURES
# =============================================================================

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error: on average, how far off (in %) are the
    predictions? MAPE of 10 means "typically about 10% wrong".
    Days where the true value is 0 are skipped (a percentage of zero is
    undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true > 0
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error: the typical size of the error in the ORIGINAL
    units (medicine units per day). It punishes big misses more than small ones.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# =============================================================================
# SECTION 4: TRAIN, EVALUATE, SAVE
# =============================================================================

def train_and_evaluate():
    # ---- Step 1: load the CSV and sort it ----
    raw = pd.read_csv(CSV_PATH, parse_dates=["date"])
    raw = raw.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    print(f"Loaded {len(raw):,} rows from {CSV_PATH}")

    # ---- Step 2: build the features ----
    df = build_features(raw)

    # ---- Step 3: encode district and drug name as numbers ----
    district_map = {name: i for i, name in enumerate(sorted(df["district"].unique()))}
    drug_map = {name: i for i, name in enumerate(sorted(df["drug_name"].unique()))}
    df = encode_categories(df, district_map, drug_map)

    # The first ~14 days of each PHC/drug have no "past" to look back on, so
    # their lag/rolling features are empty. We drop those rows.
    df = df.dropna(subset=["lag_1", "lag_7", "roll_7", "roll_14"])

    # ---- Step 4: split by time, NOT randomly ----
    # A random split would let the model "peek" at days next to the ones it is
    # tested on, which makes results look better than they would be in real
    # life. Real forecasting always means predicting the future from the past,
    # so we cut at a date: everything before trains, everything after tests.
    all_dates = np.sort(raw["date"].unique())
    split_date = all_dates[int(len(all_dates) * (1 - TEST_FRACTION))]
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows)")
    print(f"Test : {test['date'].min().date()} to {test['date'].max().date()} ({len(test):,} rows)")

    X_train, y_train = train[FEATURE_COLUMNS], train["units_consumed"]
    X_test, y_test = test[FEATURE_COLUMNS], test["units_consumed"]

    # ---- Step 5: train the LightGBM model ----
    # objective="tweedie" suits demand/count-like data: it can never predict
    # a negative number and copes well with occasional big spikes.
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,   # between 1 (count-like) and 2 (skewed positive)
        n_estimators=300,             # number of small decision trees to build
        learning_rate=0.05,           # how big a step each tree takes
        num_leaves=31,                # complexity of each tree
        random_state=42,              # reproducible results
        verbose=-1,                   # keep LightGBM quiet
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

    # ---- Step 6: evaluate on the unseen test period ----
    predictions = model.predict(X_test)
    test_mape = mape(y_test, predictions)
    test_rmse = rmse(y_test, predictions)

    # A "naive" guess (tomorrow = today) gives a yardstick. A useful model
    # should beat it or at least come close.
    naive_mape = mape(y_test, X_test["lag_1"])
    naive_rmse = rmse(y_test, X_test["lag_1"])

    print()
    print("=" * 55)
    print(" MODEL EVALUATION (test set, model has never seen it)")
    print("=" * 55)
    print(f" MAPE : {test_mape:6.2f} %      (average % error)")
    print(f" RMSE : {test_rmse:6.2f} units  (typical error per PHC-day)")
    print("-" * 55)
    print(f" Naive 'same as yesterday' baseline:")
    print(f" MAPE : {naive_mape:6.2f} %      RMSE : {naive_rmse:6.2f} units")
    print("=" * 55)
    if test_mape <= MAPE_WARNING_THRESHOLD:
        print(f" OK: MAPE is at or below {MAPE_WARNING_THRESHOLD:.0f}%, which is reasonable for this kind of data.")
    else:
        print(f" WARNING: MAPE is above {MAPE_WARNING_THRESHOLD:.0f}%. Something may be wrong. Things to check:")
        print("   - Is the CSV complete, with one row per day per PHC/drug?")
        print("   - Were the rows sorted by date before building lag features?")
        print("   - Is there a lot of zero or near-zero demand in the data?")
    print()
    print(" Note: these scores measure 1-day-ahead accuracy (the model is given the")
    print(" real previous days). The 7-day forecast from predict_demand() feeds its")
    print(" own predictions back in, so expect it to be somewhat less accurate.")
    print()

    # ---- Step 7: save the model ----
    # We save the model together with the number-codes for districts and drugs
    # and the feature list, because predict_demand() needs all of them.
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "district_map": district_map,
        "drug_map": drug_map,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved model to {MODEL_PATH}")

    return raw


# =============================================================================
# SECTION 5: THE FORECASTING FUNCTION
# =============================================================================

def predict_demand(phc_id, drug_name, recent_data, model_path=MODEL_PATH):
    """
    Forecasts the next 7 days of demand for ONE PHC and ONE drug.

    phc_id       : e.g. "PHC_03"
    drug_name    : e.g. "antimalarial"
    recent_data  : a DataFrame with the same columns as phc_consumption.csv
                   (date, phc_id, district, drug_name, units_consumed, ...),
                   holding the most recent days for that PHC/drug. It needs
                   at least 14 consecutive daily rows, ending on the latest
                   day you know about.

    Returns a list of 7 numbers: the forecast for day+1, day+2, ... day+7.

    How it works: predict tomorrow, add that prediction to the history as if
    it had really happened, then predict the day after, and so on.
    """
    # Load the saved model and its encodings
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    district_map = bundle["district_map"]
    drug_map = bundle["drug_map"]

    # Keep only this PHC's rows for this drug, in date order
    history = recent_data[(recent_data["phc_id"] == phc_id) &
                          (recent_data["drug_name"] == drug_name)].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    # Basic safety checks so the user gets a clear message, not a confusing crash
    if len(history) < MIN_HISTORY_DAYS:
        raise ValueError(f"Need at least {MIN_HISTORY_DAYS} days of data for {phc_id}/{drug_name}, "
                         f"got {len(history)}.")
    if not (history["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("recent_data must have one row per day with no missing days.")

    district = history["district"].iloc[-1]
    history = history[["date", "phc_id", "district", "drug_name", "units_consumed"]].copy()
    history["units_consumed"] = history["units_consumed"].astype(float)

    forecasts = []
    for _ in range(FORECAST_DAYS):
        # 1. Add an empty row for the next day (its demand is unknown yet)
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        new_row = pd.DataFrame({
            "date": [next_date], "phc_id": [phc_id], "district": [district],
            "drug_name": [drug_name], "units_consumed": [np.nan],
        })
        history = pd.concat([history, new_row], ignore_index=True)

        # 2. Build features with the SAME functions used in training.
        #    The empty last row gets its lags/averages from the days before it.
        features = encode_categories(build_features(history), district_map, drug_map)
        X_next = features.iloc[[-1]][feature_columns]

        # 3. Predict, then write the prediction into the history so the next
        #    loop pass can use it as "yesterday".
        prediction = float(model.predict(X_next)[0])
        history.loc[history.index[-1], "units_consumed"] = prediction
        forecasts.append(round(prediction, 2))

    return forecasts


# =============================================================================
# SECTION 6: RUN EVERYTHING
# "if __name__ == '__main__'" means: run this part only when you run this file
# directly, NOT when another script imports predict_demand from it.
# =============================================================================

if __name__ == "__main__":
    raw_data = train_and_evaluate()

    # Quick demo: forecast the next 7 days for one PHC/drug using its last 30 days
    demo_phc, demo_drug = "PHC_01", "antimalarial"
    last_30 = raw_data[(raw_data["phc_id"] == demo_phc) &
                       (raw_data["drug_name"] == demo_drug)].tail(30)
    forecast = predict_demand(demo_phc, demo_drug, last_30)
    print()
    print(f"Demo: last 7 actual days for {demo_phc} / {demo_drug}: "
          f"{last_30['units_consumed'].tail(7).tolist()}")
    print(f"Demo: next 7 days forecast: {forecast}")