"""
===============================================================================
 PHC MEDICINE SUPPLY - SYNTHETIC DATA GENERATOR
===============================================================================

WHAT THIS SCRIPT DOES
---------------------
Creates 2 years of fake-but-realistic daily medicine-consumption data for
20 Primary Health Centres (PHCs) in 5 districts, plus batch/expiry records
and a district distance matrix. It writes three CSV files:

    phc_consumption.csv   - one row per day, per PHC, per drug
    stock_batches.csv     - the batches making up each PHC's stock today
    district_distances.csv- 5x5 matrix of distances between districts (km)

It then plots total daily antimalarial demand so you can check by eye that
the monsoon peak shows up.

ASSUMPTIONS BEHIND THE DATA (THIS DATA IS SYNTHETIC)
----------------------------------------------------
Nothing here is real PHC data, and no number is copied from a published
table. The shape below is a story written from broad public-health ideas that
are loosely inspired by India's National Health Profile (NHP) 2023, published
by the Central Bureau of Health Intelligence. We have NOT checked these
points line by line against the report, so treat every one as an assumption:

  1. MALARIA IS GEOGRAPHICALLY CONCENTRATED. The NHP shows malaria cases
     clustering in a limited set of states/regions rather than being spread
     evenly. We copy that by making 2 of the 5 districts "high-burden", with
     about 3x the baseline antimalarial demand of the other 3 districts.

  2. MALARIA IS MONSOON-SEASONAL. Transmission rises with monsoon rains and
     standing water, so antimalarial demand is about 2x baseline from
     June to September.

  3. RESPIRATORY INFECTION IS THE LARGEST COMMUNICABLE-DISEASE CATEGORY BY
     CASE COUNT. Acute respiratory infections top the NHP case tables, so
     the respiratory baseline volume is set HIGHER than antimalarial.
     Respiratory demand peaks in winter (November-February).

  4. REAL DATA IS NOISY AND SPIKY. We add +/-15% random daily noise, and
     2-3 random "outbreak" events per PHC, per drug, per year (demand jumps
     2-3x for 5-10 days) to mimic local health emergencies.

IMPORTANT HONESTY NOTE: the exact numbers below (baseline units per day, 3x,
2x, 1.8x) are ILLUSTRATIVE ASSUMPTIONS. They are not values copied from NHP
tables. When you describe this project, call the data "synthetic, based on
stated assumptions", not "based on NHP data". If you need the data to be
quantitatively faithful, replace the constants in SECTION 1 with figures you
extract from the report.

HOW TO RUN
    python data_generator.py
The CSV files and the plot are saved in the same folder as this script, no
matter which folder your terminal is in.
===============================================================================
"""

from pathlib import Path           # builds file paths that work on Mac and Windows
import numpy as np                 # numbers and random-number tools
import pandas as pd                # tables (called "DataFrames")
import matplotlib.pyplot as plt    # plotting


# =============================================================================
# SECTION 1: SETTINGS
# Every number you might want to tweak lives here, so you never have to hunt
# through the code to change something.
# =============================================================================

# Save all output files next to this script, regardless of which folder the
# terminal is in when you run it.
OUT_DIR = Path(__file__).resolve().parent

SEED = 42  # Same seed = same "random" data every run. Change it for new data.
rng = np.random.default_rng(SEED)  # our random-number generator, used everywhere

# --- Time span: exactly 2 calendar years of daily data ---
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# --- Geography ---
DISTRICT_NAMES = ["District_A", "District_B", "District_C", "District_D", "District_E"]
PHCS_PER_DISTRICT = 4  # 5 districts x 4 PHCs = 20 PHCs
HIGH_BURDEN_DISTRICTS = ["District_B", "District_D"]  # the 2 malaria hotspots

# --- Drug names ---
ANTIMALARIAL = "antimalarial"
RESPIRATORY = "respiratory"
DRUGS = [ANTIMALARIAL, RESPIRATORY]
DRUG_CODES = {ANTIMALARIAL: "AM", RESPIRATORY: "RS"}  # short codes for batch IDs

# --- Baseline demand (units per day, for an average-sized PHC) ---
BASE_ANTIMALARIAL = 10         # normal (low-burden) districts
HIGH_BURDEN_MULTIPLIER = 3.0   # high-burden districts get 3x this
BASE_RESPIRATORY = 40          # deliberately higher than antimalarial

# --- Seasonal multipliers (1.0 means "no change from baseline") ---
MONSOON_MONTHS = [6, 7, 8, 9]       # June-September
MONSOON_MULTIPLIER = 2.0            # antimalarial ~2x in these months
WINTER_MONTHS = [11, 12, 1, 2]      # November-February
WINTER_MULTIPLIER = 1.8             # respiratory rises in these months

# --- Randomness ---
NOISE_FRACTION = 0.15               # each day's demand is scaled by 0.85 to 1.15
OUTBREAKS_PER_YEAR = (2, 3)         # min, max outbreaks per PHC per drug per year
OUTBREAK_LENGTH_DAYS = (5, 10)      # min, max outbreak duration
OUTBREAK_MULTIPLIER = (2.0, 3.0)    # min, max demand jump during an outbreak

# --- Stock batches ---
BATCHES_PER_PHC_DRUG = (3, 4)       # min, max batches per PHC per drug
EXPIRY_MIN_DAYS = 30                # ~1 month out
EXPIRY_MAX_DAYS = 365               # ~12 months out

# --- Simple restocking rule used to simulate the daily stock level ---
REORDER_POINT_DAYS = 14   # when stock falls below 14 days of normal demand...
TARGET_STOCK_DAYS = 45    # ...the PHC is resupplied up to 45 days of demand


# =============================================================================
# SECTION 2: BUILD THE LIST OF PHCs
# We create a small table with one row per PHC: its ID, its district, and a
# "size_factor" so that some PHCs are a bit busier than others (0.8 = 20%
# smaller than average, 1.2 = 20% bigger). This avoids 20 identical PHCs.
# =============================================================================

phc_rows = []
phc_counter = 1
for district in DISTRICT_NAMES:              # go through each district...
    for _ in range(PHCS_PER_DISTRICT):       # ...and make 4 PHCs in it
        phc_rows.append({
            "phc_id": f"PHC_{phc_counter:02d}",   # PHC_01, PHC_02, ... PHC_20
            "district": district,
            "size_factor": rng.uniform(0.8, 1.2),
        })
        phc_counter += 1

phcs = pd.DataFrame(phc_rows)


# =============================================================================
# SECTION 3: DATE-BASED HELPERS
# We build the list of dates once, then work out the seasonal multiplier for
# every single day. Because these depend only on the date (not on the PHC),
# we can compute them once and reuse them.
# =============================================================================

dates = pd.date_range(START_DATE, END_DATE, freq="D")  # one entry per day
n_days = len(dates)
months = dates.month.to_numpy()   # array like [1, 1, 1, ..., 12, 12]
years = dates.year.to_numpy()     # array like [2023, 2023, ..., 2024]

# np.isin(months, [6,7,8,9]) gives True on monsoon days and False otherwise.
# np.where(condition, a, b) picks 'a' where True and 'b' where False.
antimalarial_season = np.where(np.isin(months, MONSOON_MONTHS), MONSOON_MULTIPLIER, 1.0)
respiratory_season = np.where(np.isin(months, WINTER_MONTHS), WINTER_MULTIPLIER, 1.0)


# =============================================================================
# SECTION 4: HELPER FUNCTIONS
# Small reusable pieces. Each one does a single job.
# =============================================================================

def make_outbreak_multiplier():
    """
    Returns an array with one number per day. It is 1.0 on normal days and
    2-3 during an outbreak window. Multiplying demand by this array creates
    the spikes.
    """
    spike = np.ones(n_days)  # start with "no outbreak" everywhere

    for year in np.unique(years):                     # handle each year separately
        year_positions = np.where(years == year)[0]   # day-numbers belonging to this year
        first_day, last_day = year_positions[0], year_positions[-1]

        n_events = rng.integers(OUTBREAKS_PER_YEAR[0], OUTBREAKS_PER_YEAR[1] + 1)
        for _ in range(n_events):
            length = rng.integers(OUTBREAK_LENGTH_DAYS[0], OUTBREAK_LENGTH_DAYS[1] + 1)
            # Pick a start day so the whole window fits inside this year
            start = rng.integers(first_day, last_day - length + 2)
            size = rng.uniform(OUTBREAK_MULTIPLIER[0], OUTBREAK_MULTIPLIER[1])
            end = start + length
            # np.maximum keeps the bigger value if two outbreaks overlap,
            # so overlapping events don't multiply into something absurd.
            spike[start:end] = np.maximum(spike[start:end], size)

    return spike


def simulate_stock(daily_units, baseline_per_day):
    """
    Simulates the end-of-day stock level for one PHC and one drug.
    Each day the stock goes down by what was consumed. If it drops below the
    reorder point, a resupply tops it back up to the target level.
    Returns an array of stock levels (one per day).
    """
    reorder_point = int(round(baseline_per_day * REORDER_POINT_DAYS))
    target_stock = int(round(baseline_per_day * TARGET_STOCK_DAYS))

    stock = target_stock                       # start the 2 years fully stocked
    history = np.empty(n_days, dtype=int)      # empty array to fill in day by day

    for day in range(n_days):
        # A PHC can't hand out more than it has, so cap consumption at stock
        stock -= min(int(daily_units[day]), stock)
        if stock < reorder_point:              # running low -> resupply
            stock = target_stock
        history[day] = stock

    return history


def split_stock_into_batches(total_units, n_batches):
    """
    Splits a total quantity into n_batches whole-number pieces that add up
    exactly to total_units, with every batch holding at least 1 unit.
    """
    shares = rng.dirichlet(np.ones(n_batches))                   # random proportions summing to 1
    extra = rng.multinomial(total_units - n_batches, shares)     # hand out units by those proportions
    return extra + 1                                             # +1 guarantees no empty batch


def make_expiry_offsets(n_batches):
    """
    Returns n_batches expiry offsets (days from today), spread between
    EXPIRY_MIN_DAYS and EXPIRY_MAX_DAYS. We cut that range into equal
    bands and pick one random day per band so expiry dates are well spread
    (not all bunched together by bad luck).
    """
    edges = np.linspace(EXPIRY_MIN_DAYS, EXPIRY_MAX_DAYS, n_batches + 1).astype(int)
    return [int(rng.integers(edges[i], edges[i + 1])) for i in range(n_batches)]


# =============================================================================
# SECTION 5: GENERATE DEMAND, STOCK AND BATCHES FOR EVERY PHC x DRUG
# For each of the 20 PHCs and each of the 2 drugs we:
#   1. compute the expected daily demand (baseline x season x outbreaks),
#   2. add +/-15% noise and round to whole units,
#   3. simulate the daily stock level,
#   4. split the final stock into 3-4 batches with expiry dates.
# "Today" for expiry purposes is the last date in the data, so the batches
# describe the stock on hand at the end of the 2 years. Their quantities add
# up exactly to the final-day current_stock value.
# =============================================================================

snapshot_date = dates[-1]      # the "today" that expiry dates are measured from
consumption_frames = []        # will collect one small table per PHC x drug
batch_rows = []                # will collect one dict per batch

for _, phc in phcs.iterrows():
    is_high_burden = phc["district"] in HIGH_BURDEN_DISTRICTS

    for drug in DRUGS:
        # --- 5a. Baseline demand and seasonal shape for this drug ---
        if drug == ANTIMALARIAL:
            baseline = BASE_ANTIMALARIAL * phc["size_factor"]
            if is_high_burden:
                baseline *= HIGH_BURDEN_MULTIPLIER    # the 3x hotspot effect
            season = antimalarial_season
        else:
            baseline = BASE_RESPIRATORY * phc["size_factor"]
            season = respiratory_season

        # --- 5b. Combine baseline x season x outbreak spikes x daily noise ---
        outbreak = make_outbreak_multiplier()
        noise = rng.uniform(1 - NOISE_FRACTION, 1 + NOISE_FRACTION, size=n_days)
        expected = baseline * season * outbreak * noise
        units = np.maximum(np.rint(expected), 0).astype(int)  # whole units, never negative

        # --- 5c. Daily stock level ---
        stock = simulate_stock(units, baseline)

        consumption_frames.append(pd.DataFrame({
            "date": dates,
            "phc_id": phc["phc_id"],
            "district": phc["district"],
            "drug_name": drug,
            "units_consumed": units,
            "current_stock": stock,
        }))

        # --- 5d. Batches: split the final-day stock into 3-4 batches ---
        n_batches = int(rng.integers(BATCHES_PER_PHC_DRUG[0], BATCHES_PER_PHC_DRUG[1] + 1))
        quantities = split_stock_into_batches(int(stock[-1]), n_batches)
        expiry_offsets = sorted(make_expiry_offsets(n_batches))  # earliest expiry first

        for i in range(n_batches):
            batch_rows.append({
                "phc_id": phc["phc_id"],
                "drug_name": drug,
                "batch_id": f"{phc['phc_id']}-{DRUG_CODES[drug]}-B{i + 1}",
                "quantity": int(quantities[i]),
                "expiry_date": snapshot_date + pd.Timedelta(days=expiry_offsets[i]),
            })

# Stack the 40 small tables (20 PHCs x 2 drugs) into one big table
consumption = pd.concat(consumption_frames, ignore_index=True)
batches = pd.DataFrame(batch_rows)


# =============================================================================
# SECTION 6: DISTRICT DISTANCE MATRIX (5 x 5, in km)
# Purely random numbers would give impossible geography (A-B short, B-C short,
# but A-C huge). Instead we drop 5 random points on an imaginary map, measure
# the straight-line distances, and multiply by 1.25 because roads wind and are
# longer than straight lines. We keep trying until every pair of districts is
# between 20 and 200 km apart.
# =============================================================================

MIN_KM, MAX_KM = 20, 200
ROAD_FACTOR = 1.25          # road distance is longer than straight-line distance
MAP_SIZE_KM = 120           # districts are placed inside a 120 x 120 km box
n_districts = len(DISTRICT_NAMES)

for attempt in range(10_000):
    points = rng.uniform(0, MAP_SIZE_KM, size=(n_districts, 2))   # (x, y) per district
    diff = points[:, None, :] - points[None, :, :]                # differences between every pair
    dist = np.sqrt((diff ** 2).sum(axis=2)) * ROAD_FACTOR         # straight-line x road factor
    off_diagonal = dist[~np.eye(n_districts, dtype=bool)]         # ignore the zeros on the diagonal
    if off_diagonal.min() >= MIN_KM and off_diagonal.max() <= MAX_KM:
        break                                                     # good layout found
else:
    raise RuntimeError("Could not find a valid district layout; adjust the settings.")

distances = pd.DataFrame(dist.round(1), index=DISTRICT_NAMES, columns=DISTRICT_NAMES)


# =============================================================================
# SECTION 7: SAVE THE THREE CSV FILES
# We convert dates to plain "YYYY-MM-DD" text so the CSVs look clean.
# (We do this on copies, because the plot below still needs real dates.)
# =============================================================================

out_consumption = consumption.copy()
out_consumption["date"] = out_consumption["date"].dt.strftime("%Y-%m-%d")
out_consumption.to_csv(OUT_DIR / "phc_consumption.csv", index=False)

out_batches = batches.copy()
out_batches["expiry_date"] = out_batches["expiry_date"].dt.strftime("%Y-%m-%d")
out_batches.to_csv(OUT_DIR / "stock_batches.csv", index=False)

distances.to_csv(OUT_DIR / "district_distances.csv", index_label="district")

print(f"phc_consumption.csv    : {len(consumption):,} rows")
print(f"stock_batches.csv      : {len(batches)} rows")
print("district_distances.csv : 5x5 matrix")
print()
print("Sanity check - average antimalarial units per PHC per day, by district:")
print(consumption[consumption["drug_name"] == ANTIMALARIAL]
      .groupby("district")["units_consumed"].mean().round(1).to_string())
print()
print("Sanity check - average units per PHC per day, by drug:")
print(consumption.groupby("drug_name")["units_consumed"].mean().round(1).to_string())


# =============================================================================
# SECTION 8: PLOT TOTAL DAILY ANTIMALARIAL DEMAND
# We add up antimalarial units across all 20 PHCs for each date. If the data
# is right you should see:
#   - a big hump every June-September (shaded in the plot),
#   - jagged day-to-day noise,
#   - occasional sharp spikes from outbreaks.
# The thick line is a 7-day rolling average, which smooths the noise so the
# seasonal shape is easier to see.
# =============================================================================

antimalarial_rows = consumption[consumption["drug_name"] == ANTIMALARIAL]
daily_total = antimalarial_rows.groupby("date")["units_consumed"].sum()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily_total.index, daily_total.values, color="tab:blue", alpha=0.4,
        linewidth=0.8, label="Daily total")
ax.plot(daily_total.index, daily_total.rolling(7, center=True).mean(),
        color="tab:blue", linewidth=2, label="7-day average")

# Shade each monsoon window (June 1 - Sept 30) so the peak is easy to spot
for i, year in enumerate(np.unique(years)):
    ax.axvspan(pd.Timestamp(year, 6, 1), pd.Timestamp(year, 9, 30),
               color="tab:green", alpha=0.15,
               label="Monsoon (Jun-Sep)" if i == 0 else None)

ax.set_title("Total daily antimalarial demand across all 20 PHCs")
ax.set_xlabel("Date")
ax.set_ylabel("Units consumed per day")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "antimalarial_daily_demand.png", dpi=150)  # also save a copy as an image
plt.show()
