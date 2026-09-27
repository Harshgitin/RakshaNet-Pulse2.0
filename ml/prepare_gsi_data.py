"""
RakshaNet - GSI India Landslide Dataset Preparation

Input:
    GSI_landslide_inventory.csv

Output:
    gsi_india_cleaned.csv
    gsi_ner_cleaned.csv
    gsi_state_summary.csv

This script:
- Cleans state names
- Cleans movement types
- Cleans material names
- Parses history years where available
- Identifies NER states
- Removes invalid coordinates
"""

from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parent

INPUT_FILE = ML_DIR / "GSI_landslide_inventory.csv"

INDIA_OUTPUT = ML_DIR / "gsi_india_cleaned.csv"
NER_OUTPUT = ML_DIR / "gsi_ner_cleaned.csv"
SUMMARY_OUTPUT = ML_DIR / "gsi_state_summary.csv"


# ============================================================
# NER STATES
# ============================================================

NER_STATES = {
    "Assam",
    "Arunachal Pradesh",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura",
}


# ============================================================
# STATE NORMALIZATION
# ============================================================

def normalize_state(value):

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    replacements = {
        "KERALA": "Kerala",
        "KARNATAKA": "Karnataka",
        "Tamil nadu": "Tamil Nadu",
        "MEGHALAYA": "Meghalaya",
        "-Arunachal Pradesh": "Arunachal Pradesh",
        "Arunachal Pradesh": "Arunachal Pradesh",
        "JAMMU & KASHMIR (UT)": "Jammu & Kashmir (UT)",
        "Jammu and Kashmir": "Jammu & Kashmir (UT)",
    }

    return replacements.get(
        text,
        text
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HISTORY YEAR EXTRACTION
# ============================================================

def extract_history_year(value):

    if pd.isna(value):
        return None

    text = str(value)

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        text
    )

    if not years:
        return None

    years = [
        int(y)
        for y in years
    ]

    return min(years)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAKSHA​NET - GSI DATA PREPARATION")
    print("==============================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nLoaded records: {len(df):,}"
    )

    # --------------------------------------------------------
    # CLEAN STATE
    # --------------------------------------------------------

    df["state_normalized"] = (
        df["state"]
        .apply(normalize_state)
    )

    # --------------------------------------------------------
    # CLEAN TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "slide_no",
        "state",
        "district",
        "slide_name",
        "nh_sh_location",
        "material_involved",
        "movement_type",
        "history",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .apply(clean_text)
            )

    # --------------------------------------------------------
    # CLEAN COORDINATES
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    before = len(df)

    df = df[
        df["latitude"].between(-90, 90)
        &
        df["longitude"].between(-180, 180)
    ].copy()

    removed = before - len(df)

    print(
        f"Invalid coordinate rows removed: "
        f"{removed:,}"
    )

    # --------------------------------------------------------
    # HISTORY YEAR
    # --------------------------------------------------------

    df["history_year"] = (
        df["history"]
        .apply(extract_history_year)
    )

    # --------------------------------------------------------
    # NER FLAG
    # --------------------------------------------------------

    df["is_ner"] = (
        df["state_normalized"]
        .isin(NER_STATES)
    )

    # --------------------------------------------------------
    # STANDARDIZE MOVEMENT TYPE
    # --------------------------------------------------------

    df["movement_type_normalized"] = (
        df["movement_type"]
        .str.lower()
        .str.strip()
    )

    # Group common spelling variants.
    movement_map = {
        "slide": "Slide",
        "fall": "Fall",
        "flow": "Flow",
        "flows": "Flow",
        "subsidence": "Subsidence",
        "creep": "Creep",
        "slump": "Slump",
        "topple": "Topple",
        "composite": "Composite",
        "complex": "Complex",
        "rock slide": "Rock Slide",
        "debris slide": "Debris Slide",
        "soil slide": "Soil Slide",
    }

    df["movement_type_normalized"] = (
        df["movement_type_normalized"]
        .map(movement_map)
        .fillna(
            df["movement_type"]
            .replace("", "Unknown")
        )
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        by="sl_no"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # NER DATASET
    # --------------------------------------------------------

    ner = df[
        df["is_ner"]
    ].copy()

    # --------------------------------------------------------
    # STATE SUMMARY
    # --------------------------------------------------------

    summary = (
        df.groupby(
            "state_normalized"
        )
        .agg(
            landslide_records=(
                "sl_no",
                "count"
            ),
            unique_districts=(
                "district",
                "nunique"
            ),
            latitude_mean=(
                "latitude",
                "mean"
            ),
            longitude_mean=(
                "longitude",
                "mean"
            ),
        )
        .sort_values(
            by="landslide_records",
            ascending=False
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
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

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n==============================================")
    print("GSI PREPARATION COMPLETE")
    print("==============================================")

    print(
        f"\nIndia records: {len(df):,}"
    )

    print(
        f"NER records: {len(ner):,}"
    )

    print(
        f"\nNER state distribution:"
    )

    print(
        ner[
            "state_normalized"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFiles created:"
    )

    print(
        f"  ✅ {INDIA_OUTPUT.name}"
    )

    print(
        f"  ✅ {NER_OUTPUT.name}"
    )

    print(
        f"  ✅ {SUMMARY_OUTPUT.name}"
    )


if __name__ == "__main__":
    main()