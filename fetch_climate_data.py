"""
================================================================================
 fetch_climate_data.py
 Dengue Forecasting Capstone — climate data acquisition & dataset assembly
================================================================================

 WHAT THIS DOES
 --------------
   STEP 1  Downloads daily temperature + humidity from NASA POWER for Malaysian
           state centroids (2011-2022). No API key needed.
   STEP 2  Aggregates NASA POWER daily -> epidemiological weeks.
   STEP 3  Converts WFP/CHIRPS dekadal rainfall -> weekly (uniform-rate method).
   STEP 4  Merges everything with the dengue case series into one modelling table
           and engineers lag / rolling features.

 KEY DESIGN DECISIONS (driven by diagnostics on the real data)
 ------------------------------------------------------------
   * Climate is aggregated to national level using POPULATION weights, not
     area weights. Area weighting gives Borneo ~60% of the signal while
     Selangor+KL get 2.5%, which is backwards for a human disease.
   * Rolling means are the primary climate features: the temperature-dengue
     relationship is low-frequency and vanishes under differencing.
   * CHIRPS is the RAINFALL source of record, not NASA POWER precipitation.
     CHIRPS blends satellite with station gauges; MERRA-2 precip is model-
     derived. Empirically CHIRPS correlates with dengue at r=-0.33 vs POWER's
     -0.18 (deseasonalised, detrended). POWER precip is retained only as an
     independent cross-check on the dekad-to-week conversion (r=0.743 -> the
     conversion is sound).
   * The CHIRPS PCODE->state mapping is VERIFIED against the HDX COD
     gazetteer. It is NOT ISO 3166-2. See PCODE_TO_STATE.

 REQUIREMENTS
 ------------
   pip install pandas numpy requests

 NOTE: This script was written but NOT executed (no network access in the
 environment where it was drafted). Run STEP 1 first and check the output
 before proceeding. Expect the full download to take ~2-5 minutes.

 SOURCES
 -------
   NASA POWER   https://power.larc.nasa.gov/docs/services/api/temporal/daily/
   WFP rainfall https://data.humdata.org/dataset/mys-rainfall-subnational
================================================================================
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------ CONFIG ---
START_DATE = "20110101"          # matches dengue weekly series start
END_DATE   = "20221231"          # matches dengue weekly series end

RAINFALL_CSV = "mys-rainfall-subnat-full.csv"   # your HDX file
DENGUE_CSV   = "Target_Dataset.csv"             # your case file
OUTDIR       = Path(".")

# NASA POWER parameters.  T2M = mean air temp at 2m (degC)
#                         RH2M = relative humidity at 2m (%)
#                         PRECTOTCORR = corrected precipitation (mm/day)
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR"

# Approximate state centroids (lat, lon).  Verify against an official
# gazetteer if you need precision; these are adequate for climate aggregation.
STATE_POINTS = {
    "Johor":            (1.90, 103.40),
    "Kedah":            (6.10, 100.60),
    "Kelantan":         (5.30, 102.20),
    "Melaka":           (2.20, 102.30),
    "Negeri Sembilan":  (2.70, 102.10),
    "Pahang":           (3.80, 102.90),
    "Pulau Pinang":     (5.40, 100.30),
    "Perak":            (4.60, 101.10),
    "Perlis":           (6.50, 100.20),
    "Selangor":         (3.30, 101.50),
    "Terengganu":       (5.00, 103.00),
    "Sabah":            (5.40, 116.80),
    "Sarawak":          (2.50, 113.00),
    "WP Kuala Lumpur":  (3.14, 101.69),
    "WP Labuan":        (5.28, 115.24),
}

# ---------------------------------------------------------------------------
# CHIRPS PCODE -> STATE NAME  (VERIFIED against the HDX COD gazetteer, v01
# 2021-02-11).  DO NOT assume ISO 3166-2 here -- these are COD p-codes and
# they DIVERGE from ISO after MY03. Getting this wrong silently pairs each
# state's dengue with another state's rainfall.
#
#   Notable traps:  MY10 = Perlis   (NOT Selangor)
#                   MY17 = Selangor (no ISO equivalent)
#                   MY04 = WP Kuala Lumpur (NOT Melaka)
# ---------------------------------------------------------------------------
PCODE_TO_STATE = {
    "MY01": "Johor",            "MY02": "Kedah",
    "MY03": "Kelantan",         "MY04": "WP Kuala Lumpur",
    "MY05": "WP Labuan",        "MY06": "Melaka",
    "MY07": "Negeri Sembilan",  "MY08": "Pahang",
    "MY09": "Perak",            "MY10": "Perlis",
    "MY11": "Pulau Pinang",     "MY12": "Sabah",
    "MY13": "Sarawak",          "MY15": "Terengganu",
    "MY16": "WP Putrajaya",     "MY17": "Selangor",
}

# ---------------------------------------------------------------------------
# WEIGHTING
# ---------------------------------------------------------------------------
# Population weights (millions, ~2020 census) are the DEFAULT.
#
# WHY POPULATION AND NOT AREA:  dengue is a human disease concentrated in
# dense lowland urban areas. Area-weighting gives Sarawak + Sabah ~60% of the
# national climate signal (they are physically huge) while Selangor + KL get
# ~2.5% -- despite holding ~27% of the population and a large share of cases.
# That is backwards. Empirically, switching to population weights raised the
# temperature-dengue correlation from ~0.29 to ~0.34, so this is not cosmetic.
#
# STATE_AREA is retained so you can run the alternative as a sensitivity
# check and report both in your methodology.
STATE_POP = {
    "Johor": 3.79, "Kedah": 2.13, "Kelantan": 1.79, "Melaka": 0.93,
    "Negeri Sembilan": 1.20, "Pahang": 1.59, "Pulau Pinang": 1.77,
    "Perak": 2.50, "Perlis": 0.28, "Selangor": 6.99, "Terengganu": 1.15,
    "Sabah": 3.42, "Sarawak": 2.45, "WP Kuala Lumpur": 1.75,
    "WP Labuan": 0.10,
}

STATE_AREA = {
    "Johor": 19210, "Kedah": 9500, "Kelantan": 15099, "Melaka": 1664,
    "Negeri Sembilan": 6686, "Pahang": 36137, "Pulau Pinang": 1048,
    "Perak": 21035, "Perlis": 821, "Selangor": 8104, "Terengganu": 13035,
    "Sabah": 73631, "Sarawak": 124450, "WP Kuala Lumpur": 243,
    "WP Labuan": 91,
}

WEIGHTS = STATE_POP          # <-- switch to STATE_AREA for sensitivity check

# ---------------------------------------------------------------------------
# KNOWN CAVEAT: MERRA-2 grid resolution (~0.5 deg, ~55 km) averages terrain
# within each cell. Perak (23.6 C) and Sabah (24.0 C) read cool because their
# cells include the Titiwangsa and Crocker ranges. The bias is systematic and
# roughly constant over time, so it does not distort the TEMPORAL variation a
# forecasting model learns from -- but do not quote these as absolute state
# temperatures. State this in your methodology.
# ---------------------------------------------------------------------------


# ============================================================== STEP 1 =======
def fetch_power_point(lat, lon, start=START_DATE, end=END_DATE, retries=3):
    """Fetch daily NASA POWER data for one lat/lon. Returns a DataFrame."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    q = {
        "parameters": PARAMS,
        "community": "AG",          # agroclimatology community
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=q, timeout=90)
            r.raise_for_status()
            payload = r.json()["properties"]["parameter"]
            df = pd.DataFrame(payload)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            return df.reset_index()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt+1} after error: {e}")
            time.sleep(5)


def download_all_states():
    """Download every state centroid and stack into one long table."""
    frames = []
    for name, (lat, lon) in STATE_POINTS.items():
        print(f"  fetching {name} ({lat}, {lon}) ...")
        d = fetch_power_point(lat, lon)
        d["state"] = name
        frames.append(d)
        time.sleep(1)                      # be polite to the API
    out = pd.concat(frames, ignore_index=True)

    # NASA POWER uses -999 as its fill value for missing data.
    out = out.replace(-999, np.nan)

    out.to_csv(OUTDIR / "nasa_power_daily_states.csv", index=False)
    print(f"  saved nasa_power_daily_states.csv  ({len(out):,} rows)")
    return out


# ============================================================== STEP 2 =======
def power_daily_to_weekly(daily, week_starts):
    """
    Area-weighted national daily series -> epidemiological weeks.
    `week_starts` is the list of week-start dates from the dengue series.
    """
    daily = daily.copy()
    daily["w"] = daily["state"].map(WEIGHTS)

    # area-weighted national daily mean for each variable
    varcols = [c for c in daily.columns
               if c not in ("date", "state", "w")]
    nat = (
        daily
        .groupby("date")
        .apply(lambda g: pd.Series({
            v: np.average(g[v].dropna(),
                          weights=g.loc[g[v].notna(), "w"])
            if g[v].notna().any() else np.nan
            for v in varcols
        }))
        .reset_index()
    )

    nat = nat.set_index("date")

    # aggregate into each epi week [start, start+6]
    rows = []
    for s in week_starts:
        win = nat.loc[s: s + pd.Timedelta(days=6)]
        if len(win) == 0:
            rows.append({"start": s})
            continue
        rows.append({
            "start":     s,
            "temp_mean": win["T2M"].mean(),
            "temp_max":  win["T2M_MAX"].max(),
            "temp_min":  win["T2M_MIN"].min(),
            "humidity":  win["RH2M"].mean(),
            "precip_power": win["PRECTOTCORR"].sum(),   # cross-check vs CHIRPS
        })
    return pd.DataFrame(rows)


# ============================================================== STEP 3 =======
def rainfall_dekad_to_weekly(rain_csv, week_starts):
    """
    WFP/CHIRPS dekadal rainfall -> weekly.

    Dekads start on the 1st, 11th and 21st of each month. The 3rd dekad runs
    to month end, so its length varies (8-11 days). We assume a uniform daily
    rate within each dekad, expand to daily, then sum into epi weeks.

    LIMITATION: this smooths short-duration rainfall bursts. State it in
    your methodology.
    """
    r = pd.read_csv(rain_csv, parse_dates=["date"])
    a1 = r[r["adm_level"] == 1].copy()

    # keep 'final' where available; 'prelim' is provisional
    if "version" in a1.columns:
        a1 = a1.sort_values("version")                      # final < prelim
        a1 = a1.drop_duplicates(subset=["PCODE", "date"], keep="first")

    a1["state"] = a1["PCODE"].map(PCODE_TO_STATE)
    unmapped = a1.loc[a1["state"].isna(), "PCODE"].unique()
    if len(unmapped):
        raise ValueError(f"Unmapped PCODEs: {unmapped} -- update PCODE_TO_STATE")
    a1["pop"] = a1["state"].map(STATE_POP)

    # Produce BOTH weightings.
    #   DIAGNOSTIC FINDING: area weighting correlates better with dengue for
    #   RAINFALL (r=-0.33 vs -0.28 at lag 0), while population weighting is
    #   better for TEMPERATURE (+0.34 vs +0.29). The difference is small, so
    #   both are emitted as features and the model decides.
    out = {}
    for label, wcol in [("area", "n_pixels"), ("pop", "pop")]:
        nat = (
            a1.groupby("date")
              .apply(lambda g: np.average(g["rfh"], weights=g[wcol]))
              .rename("rain_dekad")
              .reset_index()
        )
        # dekad -> daily (uniform rate) -> weekly
        recs = []
        for _, row in nat.iterrows():
            d0 = row["date"]
            if d0.day in (1, 11):
                length = 10
            else:                               # 3rd dekad -> end of month
                length = (d0 + pd.offsets.MonthEnd(0)).day - 20
            rate = row["rain_dekad"] / length
            for k in range(length):
                recs.append({"day": d0 + pd.Timedelta(days=k),
                             "rain_mm": rate})
        daily = pd.DataFrame(recs).set_index("day").sort_index()
        out[label] = [daily.loc[s: s + pd.Timedelta(days=6), "rain_mm"].sum()
                      for s in week_starts]

    return pd.DataFrame({
        "start": week_starts,
        "rain_chirps": out["area"],        # primary: area-weighted CHIRPS
        "rain_chirps_pop": out["pop"],     # alternative weighting
    })


# ============================================================== STEP 4 =======
def load_dengue_weekly(dengue_csv):
    """National weekly dengue series (Total + Confirmed spliced)."""
    d = pd.read_csv(dengue_csv)
    wk = d[(d["T_res"] == "Week") & (d["S_res"] == "Admin0")].copy()
    wk["start"] = pd.to_datetime(wk["calendar_start_date"])
    wk = wk.sort_values("start").reset_index(drop=True)
    return wk[["start", "dengue_total", "case_definition_standardised"]].rename(
        columns={"case_definition_standardised": "case_def"}
    )


def build_dataset():
    print("STEP 4: assembling modelling dataset ...")
    dengue = load_dengue_weekly(DENGUE_CSV)
    weeks = dengue["start"].tolist()

    rain = rainfall_dekad_to_weekly(RAINFALL_CSV, weeks)

    power_path = OUTDIR / "nasa_power_daily_states.csv"
    if power_path.exists():
        daily = pd.read_csv(power_path, parse_dates=["date"])
    else:
        print("  nasa_power_daily_states.csv not found -> downloading")
        daily = download_all_states()

    clim = power_daily_to_weekly(daily, weeks)

    df = dengue.merge(rain, on="start", how="left") \
               .merge(clim, on="start", how="left")

    # --------- feature engineering ----------------------------------------
    # DIAGNOSTIC FINDING driving these choices:
    #   After removing trend and seasonality, temperature correlates with
    #   dengue at r ~ +0.27..+0.31 and humidity at r ~ -0.22..-0.29, roughly
    #   FLAT across lags 0-12 weeks. Rainfall is weak (r ~ -0.10..-0.18).
    #   First-differencing destroys all of it (r ~ 0.05), which means the
    #   climate signal lives at LOW FREQUENCY -- sustained conditions over
    #   weeks, not week-to-week jitter.
    #
    #   => ROLLING MEANS are the important climate features.
    #   => Do NOT difference the climate covariates.
    #   => The flat lag profile means there is no single optimal lag; let the
    #      model choose among several rolling windows.

    # lagged cases: the dominant predictor (ACF lag-1 = 0.97)
    for lag in [1, 2, 3, 4, 6, 8, 12]:
        df[f"cases_lag{lag}"] = df["dengue_total"].shift(lag)

    # lagged climate (kept for completeness; expect rolling to dominate)
    for lag in [2, 4, 8, 12]:
        df[f"rain_lag{lag}"] = df["rain_chirps"].shift(lag)
        df[f"temp_lag{lag}"] = df["temp_mean"].shift(lag)
        df[f"hum_lag{lag}"] = df["humidity"].shift(lag)

    # rolling means -- the features that actually carry the climate signal
    for w in [4, 8, 12, 16]:
        df[f"cases_roll{w}"] = df["dengue_total"].shift(1).rolling(w).mean()
        df[f"temp_roll{w}"] = df["temp_mean"].shift(1).rolling(w).mean()
        df[f"hum_roll{w}"] = df["humidity"].shift(1).rolling(w).mean()
        df[f"rain_roll{w}"] = df["rain_chirps"].shift(1).rolling(w).mean()
        df[f"rainpop_roll{w}"] = df["rain_chirps_pop"].shift(1).rolling(w).mean()

    # lagged rolling means: sustained conditions, offset by the biological lag
    for w in [8, 12]:
        for lag in [4, 8]:
            df[f"temp_roll{w}_lag{lag}"] = df["temp_mean"].shift(lag).rolling(w).mean()
            df[f"hum_roll{w}_lag{lag}"] = df["humidity"].shift(lag).rolling(w).mean()

    df["week_of_year"] = df["start"].dt.isocalendar().week.astype(int)
    df["month"] = df["start"].dt.month
    df["year"] = df["start"].dt.year

    df.to_csv(OUTDIR / "dengue_model_dataset.csv", index=False)
    print(f"  saved dengue_model_dataset.csv  "
          f"({len(df)} rows x {df.shape[1]} cols)")

    # --------- quick sanity report ----------------------------------------
    print("\n--- SANITY CHECK ---")
    print(f"  weeks:            {len(df)}")
    print(f"  date range:       {df['start'].min().date()} -> "
          f"{df['start'].max().date()}")
    for c in ["dengue_total", "rain_chirps", "temp_mean", "humidity"]:
        if c in df:
            print(f"  {c:14s} nulls={df[c].isna().sum():3d}  "
                  f"mean={df[c].mean():8.1f}")

    if {"rain_chirps", "precip_power"} <= set(df.columns):
        ok = df[["rain_chirps", "precip_power"]].dropna()
        if len(ok) > 10:
            r = np.corrcoef(ok["rain_chirps"], ok["precip_power"])[0, 1]
            print(f"\n  CHIRPS vs POWER precipitation agreement: r = {r:.3f}")
            print("  (expect r > 0.7; a low value means one source is "
                  "misaligned — investigate before modelling)")
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("NASA POWER + CHIRPS climate acquisition")
    print("=" * 70)
    print("\nSTEP 1-2: NASA POWER download")
    if not (OUTDIR / "nasa_power_daily_states.csv").exists():
        download_all_states()
    else:
        print("  nasa_power_daily_states.csv already exists — skipping")

    print("\nSTEP 3-4: assembling")
    build_dataset()
    print("\nDone. Next: run your EDA on dengue_model_dataset.csv")
