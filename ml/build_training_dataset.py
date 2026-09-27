import hashlib
import json
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent

NASA_FILE = BASE_DIR / "india_landslides_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "training_dataset.csv"
MAX_POSITIVE_EVENTS = 600
CACHE_DIR = BASE_DIR / "weather_cache"

CACHE_DIR.mkdir(exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

BATCH_SIZE = 10
REQUEST_SLEEP = 3
MAX_RETRIES = 3

# Weather window required for 7-day rainfall.
WINDOW_BEFORE = 7


# ============================================================
# WEATHER API
# ============================================================

def make_cache_key(latitudes, longitudes, start_date, end_date):

    raw = (
        ",".join(f"{x:.4f}" for x in latitudes)
        + "|"
        + ",".join(f"{x:.4f}" for x in longitudes)
        + "|"
        + start_date
        + "|"
        + end_date
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


def request_weather(
    latitudes,
    longitudes,
    start_date,
    end_date
):

    key = make_cache_key(
        latitudes,
        longitudes,
        start_date,
        end_date
    )

    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    params = {
        "latitude": ",".join(
            f"{x:.4f}"
            for x in latitudes
        ),

        "longitude": ",".join(
            f"{x:.4f}"
            for x in longitudes
        ),

        "start_date": start_date,
        "end_date": end_date,

        "daily": (
            "temperature_2m_mean,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "rain_sum,"
            "precipitation_hours,"
            "wind_speed_10m_max"
        ),

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",

        "timezone": "UTC",

        # IMPORTANT:
        # ERA5 instead of ERA5-Land because
        # we need rainfall + wind variables.
        "models": "era5"
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"      API request "
                f"{attempt}/{MAX_RETRIES} "
                f"({len(latitudes)} locations)"
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=120
            )

            if response.status_code == 429:

                wait = 30 * attempt

                print(
                    f"      Rate limited. "
                    f"Waiting {wait}s..."
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            data = response.json()

            with open(
                cache_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f
                )

            time.sleep(
                REQUEST_SLEEP
            )

            return data

        except Exception as e:

            print(
                f"      ERROR: {e}"
            )

            if attempt < MAX_RETRIES:

                wait = 15 * attempt

                print(
                    f"      Waiting {wait}s..."
                )

                time.sleep(wait)

    return None


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(x):

    try:

        value = float(x)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# LOAD NASA
# ============================================================

print("=" * 70)
print("RakshaNet - Final Historical Weather Dataset Builder")
print("=" * 70)


df = pd.read_csv(NASA_FILE)


required_columns = [
    "event_date",
    "latitude",
    "longitude",
    "state_normalized",
    "is_ner"
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "event_date",
        "latitude",
        "longitude"
    ]
).copy()


# India bounds
df = df[
    (df["latitude"] >= 6) &
    (df["latitude"] <= 38) &
    (df["longitude"] >= 68) &
    (df["longitude"] <= 98)
].copy()

if len(df) > MAX_POSITIVE_EVENTS:
    # Keep a reproducible, location-aware sample for weather enrichment.
    # The full GSI/NASA master inventories remain preserved for susceptibility context.
    working = df.copy()
    grouped = working.groupby("state_normalized", dropna=False)
    per_state = max(1, MAX_POSITIVE_EVENTS // max(1, grouped.ngroups))
    sampled_parts = []
    sampled_indices = set()
    for _, group in grouped:
        part = group.sample(n=min(len(group), per_state), random_state=42)
        sampled_parts.append(part)
        sampled_indices.update(part.index.tolist())
    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else working.sample(MAX_POSITIVE_EVENTS, random_state=42)
    if len(sampled) < MAX_POSITIVE_EVENTS:
        remaining = working.loc[~working.index.isin(sampled_indices)]
        if not remaining.empty:
            extra = remaining.sample(n=min(MAX_POSITIVE_EVENTS - len(sampled), len(remaining)), random_state=42)
            sampled = pd.concat([sampled, extra], ignore_index=False)
    df = sampled.reset_index(drop=True)


df["event_date"] = (
    df["event_date"]
    .dt.normalize()
)


df["state_normalized"] = (
    df["state_normalized"]
    .fillna("Unknown")
)


df = df.reset_index(
    drop=True
)


print()
print(
    f"NASA India events: {len(df)}"
)

print(
    f"NER events: "
    f"{df['is_ner'].sum()}"
)


# ============================================================
# CREATE EVENT + CONTROL SAMPLE DATES
# ============================================================

print()
print(
    "Preparing event and control weather windows..."
)


sample_requests = []


CONTROL_OFFSETS = [
    21,
    -21,
    42,
    -42
]


for idx, event in df.iterrows():

    event_date = event["event_date"]

    # --------------------------------------------------------
    # Positive event
    # --------------------------------------------------------

    sample_requests.append({
        "sample_index": idx,
        "target": 1,
        "sample_type": "landslide_event",
        "sample_date": event_date,
        "latitude": event["latitude"],
        "longitude": event["longitude"],
        "state": event["state_normalized"],
        "is_ner": bool(event["is_ner"]),
        "original_event_date": event_date
    })


    # --------------------------------------------------------
    # Control dates
    # --------------------------------------------------------

    selected_control = None

    for offset in CONTROL_OFFSETS:

        candidate = (
            event_date +
            timedelta(days=offset)
        )

        # Same month provides seasonal matching.
        if candidate.month != event_date.month:
            continue

        if candidate.year < 2007:
            continue

        if candidate.year > 2016:
            continue

        selected_control = candidate

        break


    if selected_control is not None:

        sample_requests.append({
            "sample_index": idx,
            "target": 0,
            "sample_type": "matched_control",
            "sample_date": selected_control,
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "state": event["state_normalized"],
            "is_ner": bool(event["is_ner"]),
            "original_event_date": event_date
        })


samples = pd.DataFrame(
    sample_requests
)


print(
    f"Total samples planned: "
    f"{len(samples)}"
)

print(
    samples["target"]
    .value_counts()
)


# ============================================================
# GRID COORDINATES
# ============================================================

samples["grid_lat"] = (
    samples["latitude"]
    .round(1)
)

samples["grid_lon"] = (
    samples["longitude"]
    .round(1)
)


# ============================================================
# GROUP WEATHER REQUESTS BY DATE
# ============================================================

groups = (
    samples
    .groupby("sample_date")
)


weather = {}


all_dates = sorted(
    groups.groups.keys()
)


print()
print(
    f"Unique sample dates: "
    f"{len(all_dates)}"
)


# ============================================================
# DOWNLOAD WEATHER
# ============================================================

for count, sample_date in enumerate(
    all_dates,
    start=1
):

    group = groups.get_group(
        sample_date
    )

    locations = (
        group[
            ["grid_lat", "grid_lon"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


    start_date = (
        sample_date -
        timedelta(days=WINDOW_BEFORE)
    ).strftime("%Y-%m-%d")


    end_date = sample_date.strftime(
        "%Y-%m-%d"
    )


    print()
    print(
        f"[{count}/{len(all_dates)}] "
        f"{sample_date.date()} | "
        f"{len(locations)} locations | "
        f"{start_date} -> {end_date}"
    )


    for batch_start in range(
        0,
        len(locations),
        BATCH_SIZE
    ):

        batch = locations.iloc[
            batch_start:
            batch_start + BATCH_SIZE
        ]


        lats = batch[
            "grid_lat"
        ].tolist()

        lons = batch[
            "grid_lon"
        ].tolist()


        data = request_weather(
            lats,
            lons,
            start_date,
            end_date
        )


        if data is None:

            print(
                "      FAILED batch"
            )

            continue


        responses = (
            data
            if isinstance(data, list)
            else [data]
        )


        for item in responses:

            if (
                "latitude" not in item
                or
                "longitude" not in item
            ):

                continue


            daily = item.get(
                "daily"
            )


            if not daily:
                continue


            rlat = round(
                float(item["latitude"]),
                1
            )

            rlon = round(
                float(item["longitude"]),
                1
            )


            times = daily.get(
                "time",
                []
            )


            for i, date_text in enumerate(
                times
            ):

                current_date = (
                    pd.to_datetime(
                        date_text
                    )
                    .normalize()
                )


                def get_value(
                    name
                ):

                    values = daily.get(
                        name,
                        []
                    )

                    if i >= len(values):
                        return np.nan

                    return safe_float(
                        values[i]
                    )


                weather[
                    (
                        rlat,
                        rlon,
                        current_date
                    )
                ] = {

                    "rain":
                        get_value(
                            "rain_sum"
                        ),

                    "precipitation":
                        get_value(
                            "precipitation_sum"
                        ),

                    "precip_hours":
                        get_value(
                            "precipitation_hours"
                        ),

                    "temp_mean":
                        get_value(
                            "temperature_2m_mean"
                        ),

                    "temp_max":
                        get_value(
                            "temperature_2m_max"
                        ),

                    "temp_min":
                        get_value(
                            "temperature_2m_min"
                        ),

                    "wind_max":
                        get_value(
                            "wind_speed_10m_max"
                        )
                }


print()
print(
    f"Weather observations collected: "
    f"{len(weather)}"
)


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    lat,
    lon,
    target_date
):

    records = []


    for days_back in range(
        0,
        7
    ):

        current_date = (
            target_date -
            timedelta(days=days_back)
        )


        key = (
            round(lat, 1),
            round(lon, 1),
            current_date
        )


        record = weather.get(
            key
        )


        if record is not None:

            records.append(
                (
                    current_date,
                    record
                )
            )


    if not records:
        return None


    records = sorted(
        records,
        key=lambda x: x[0],
        reverse=True
    )


    today_key = (
        round(lat, 1),
        round(lon, 1),
        target_date
    )


    today = weather.get(
        today_key
    )


    if today is None:
        return None


    rain_values = [
        x[1]["rain"]
        for x in records
    ]

    precip_values = [
        x[1]["precipitation"]
        for x in records
    ]

    hours_values = [
        x[1]["precip_hours"]
        for x in records
    ]


    rain_3 = np.sum(
        rain_values[:3]
    )

    rain_7 = np.sum(
        rain_values[:7]
    )

    precip_3 = np.sum(
        precip_values[:3]
    )

    precip_7 = np.sum(
        precip_values[:7]
    )

    hours_3 = np.sum(
        hours_values[:3]
    )

    hours_7 = np.sum(
        hours_values[:7]
    )


    return {

        "rain_1d":
            today["rain"],

        "rain_3d":
            rain_3,

        "rain_7d":
            rain_7,

        "precip_1d":
            today["precipitation"],

        "precip_3d":
            precip_3,

        "precip_7d":
            precip_7,

        "precip_hours_1d":
            today["precip_hours"],

        "precip_hours_3d":
            hours_3,

        "precip_hours_7d":
            hours_7,

        "temp_mean":
            today["temp_mean"],

        "temp_max":
            today["temp_max"],

        "temp_min":
            today["temp_min"],

        "wind_max":
            today["wind_max"]
    }


# ============================================================
# BUILD FINAL DATASET
# ============================================================

print()
print(
    "Building training samples..."
)


final_rows = []


for idx, sample in samples.iterrows():

    features = create_features(
        sample["grid_lat"],
        sample["grid_lon"],
        sample["sample_date"]
    )


    if features is None:
        continue


    final_rows.append({

        "sample_id":
            (
                f"{'POS' if sample['target'] == 1 else 'CTRL'}"
                f"_{idx:05d}"
            ),

        "target":
            sample["target"],

        "sample_type":
            sample["sample_type"],

        "event_date":
            sample["sample_date"],

        "latitude":
            sample["latitude"],

        "longitude":
            sample["longitude"],

        "state":
            sample["state"],

        "is_ner":
            sample["is_ner"],

        "month":
            sample["sample_date"].month,

        "year":
            sample["sample_date"].year,

        **features
    })


training_df = pd.DataFrame(
    final_rows
)


# ============================================================
# CLEAN
# ============================================================

numeric_columns = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_hours_1d",
    "precip_hours_3d",
    "precip_hours_7d",
    "temp_mean",
    "temp_max",
    "temp_min",
    "wind_max",
    "latitude",
    "longitude",
    "month"
]


for column in numeric_columns:

    training_df[column] = pd.to_numeric(
        training_df[column],
        errors="coerce"
    )


training_df = training_df.dropna(
    subset=[
        "rain_1d",
        "rain_3d",
        "rain_7d",
        "temp_mean",
        "wind_max"
    ]
).copy()


training_df = (
    training_df
    .sort_values("event_date")
    .reset_index(drop=True)
)


# ============================================================
# SAVE
# ============================================================

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("TRAINING DATASET COMPLETE")
print("=" * 70)


print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(training_df)}"
)


print()
print("Targets:")

print(
    training_df["target"]
    .value_counts()
)


print()
print("Sample types:")

print(
    training_df["sample_type"]
    .value_counts()
)


print()
print("NER by target:")

print(
    training_df
    .groupby("target")["is_ner"]
    .sum()
)


print()
print("Missing values:")

print(
    training_df.isna().sum()
)


print()
print("First 10 rows:")

print(
    training_df.head(10)
    .to_string(index=False)
)


print()
print("SUCCESS.")