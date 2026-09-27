"""
RakshaNet - India Landslide Data Preparation

Purpose:
1. Load the NASA Global Landslide Catalog CSV
2. Keep only India records
3. Normalize Indian state/region names
4. Identify North Eastern Region (NER)
5. Clean dates, coordinates, and useful fields
6. Save cleaned datasets for the next ML stage

IMPORTANT:
This script prepares historical landslide EVENT data.
It does not train the final prediction model yet.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parent

INPUT_FILE = ML_DIR / "Global_Landslide_Catalog_Export_rows.csv"

INDIA_OUTPUT = ML_DIR / "india_landslides_cleaned.csv"
NER_OUTPUT = ML_DIR / "ner_landslides_cleaned.csv"
SUMMARY_OUTPUT = ML_DIR / "india_landslide_summary.csv"


# ============================================================
# 2. NER STATE DEFINITIONS
# ============================================================

NER_STATES = {
    "assam",
    "arunachal pradesh",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "sikkim",
    "tripura",
}


# ============================================================
# 3. STATE NAME NORMALIZATION
# ============================================================

def normalize_state(value):
    """
    Convert different spellings/diacritics into one standard name.
    """

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    replacements = {
        # Arunachal Pradesh
        "arunāchal pradesh": "Arunachal Pradesh",
        "arunachal pradesh": "Arunachal Pradesh",

        # Nagaland
        "nāgāland": "Nagaland",
        "nagaland": "Nagaland",

        # Meghalaya
        "meghālaya": "Meghalaya",
        "meghalaya": "Meghalaya",

        # Tamil Nadu
        "tamil nādu": "Tamil Nadu",
        "tamil nadu": "Tamil Nadu",

        # West Bengal sometimes represented as Bengal
        "bengal": "West Bengal",
        "west bengal": "West Bengal",

        # Other common India naming
        "jammu and kashmir": "Jammu and Kashmir",
        "kashmir": "Jammu and Kashmir",
    }

    return replacements.get(text, text.title())


# ============================================================
# 4. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCSV file not found:\n{INPUT_FILE}\n\n"
        "Make sure Global_Landslide_Catalog_Export_rows.csv "
        "is inside the ml folder."
    )

print("\n==============================================")
print("RAKSHANET LANDSLIDE DATA PREPARATION")
print("==============================================")

print(f"\nLoading:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset shape: {df.shape}")
print(f"Original records: {len(df):,}")


# ============================================================
# 5. BASIC COLUMN CHECK
# ============================================================

required_columns = [
    "country_code",
    "country_name",
    "admin_division_name",
    "event_date",
    "latitude",
    "longitude",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "landslide_setting",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing expected columns: {missing_columns}"
    )


# ============================================================
# 6. KEEP INDIA ONLY
# ============================================================

df["country_code"] = (
    df["country_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)

india = df[df["country_code"] == "IN"].copy()

print(f"\nIndia records: {len(india):,}")


# ============================================================
# 7. NORMALIZE STATE NAMES
# ============================================================

india["state_normalized"] = (
    india["admin_division_name"]
    .apply(normalize_state)
)


# ============================================================
# 8. CREATE NER FLAG
# ============================================================

india["is_ner"] = (
    india["state_normalized"]
    .str.lower()
    .isin(NER_STATES)
)


# ============================================================
# 9. CLEAN DATE
# ============================================================

india["event_date"] = pd.to_datetime(
    india["event_date"],
    errors="coerce"
)

india["event_year"] = india["event_date"].dt.year
india["event_month"] = india["event_date"].dt.month
india["event_day_of_year"] = (
    india["event_date"].dt.dayofyear
)


# ============================================================
# 10. CLEAN COORDINATES
# ============================================================

india["latitude"] = pd.to_numeric(
    india["latitude"],
    errors="coerce"
)

india["longitude"] = pd.to_numeric(
    india["longitude"],
    errors="coerce"
)

before_coord_filter = len(india)

india = india[
    india["latitude"].between(-90, 90)
    & india["longitude"].between(-180, 180)
].copy()

removed_coords = before_coord_filter - len(india)

print(f"Invalid coordinate records removed: {removed_coords:,}")


# ============================================================
# 11. CLEAN NUMERIC OUTCOME FIELDS
# ============================================================

for col in [
    "fatality_count",
    "injury_count",
    "gazeteer_distance",
]:
    if col in india.columns:
        india[col] = pd.to_numeric(
            india[col],
            errors="coerce"
        )


# ============================================================
# 12. KEEP USEFUL COLUMNS
# ============================================================

keep_columns = [
    "event_id",
    "event_date",
    "event_year",
    "event_month",
    "event_day_of_year",
    "country_name",
    "country_code",
    "admin_division_name",
    "state_normalized",
    "is_ner",
    "latitude",
    "longitude",
    "landslide_category",
    "landslide_trigger",
    "landslide_size",
    "landslide_setting",
    "fatality_count",
    "injury_count",
    "event_title",
    "location_description",
]

cleaned = india[
    [col for col in keep_columns if col in india.columns]
].copy()


# ============================================================
# 13. SORT BY EVENT DATE
# ============================================================

cleaned = cleaned.sort_values(
    by="event_date",
    ascending=True,
    na_position="last"
).reset_index(drop=True)


# ============================================================
# 14. NER DATASET
# ============================================================

ner = cleaned[
    cleaned["is_ner"] == True
].copy()


# ============================================================
# 15. STATE SUMMARY
# ============================================================

summary = (
    cleaned.groupby("state_normalized")
    .agg(
        landslide_events=("event_id", "count"),
        first_event=("event_date", "min"),
        last_event=("event_date", "max"),
        average_latitude=("latitude", "mean"),
        average_longitude=("longitude", "mean"),
    )
    .sort_values(
        by="landslide_events",
        ascending=False
    )
    .reset_index()
)


# ============================================================
# 16. SAVE FILES
# ============================================================

cleaned.to_csv(
    INDIA_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

ner.to_csv(
    NER_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n==============================================")
print("PREPARATION COMPLETE")
print("==============================================")

print(f"\nIndia cleaned records : {len(cleaned):,}")
print(f"NER cleaned records   : {len(ner):,}")

print("\nIndia state distribution:")
print(
    summary[
        ["state_normalized", "landslide_events"]
    ].to_string(index=False)
)

print("\nNER distribution:")
print(
    ner["state_normalized"]
    .value_counts()
    .sort_values(ascending=False)
    .to_string()
)

print("\nFiles created:")

print(f"  ✅ {INDIA_OUTPUT.name}")
print(f"  ✅ {NER_OUTPUT.name}")
print(f"  ✅ {SUMMARY_OUTPUT.name}")

print("\nNext step:")
print("We will inspect these cleaned datasets before training any model.")
print("No .pkl model has been created yet.")