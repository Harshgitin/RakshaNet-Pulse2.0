"""
RakshaNet - Combine India Landslide Inventories

Primary source:
    GSI field-validated inventory

Additional source:
    NASA Global Landslide Catalog (India subset)

Goal:
    Keep GSI as the primary inventory and add NASA events
    only when they are not likely duplicates of an existing
    GSI record.

Duplicate rule:
    A NASA record is considered a likely duplicate when:
      - it is within ~1 km of a GSI record, AND
      - if both years are known, the years are within 2 years.

This is a conservative spatial matching rule.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except ImportError:
    raise ImportError(
        "\nscikit-learn is required.\n"
        "Run:\n"
        "python -m pip install scikit-learn"
    )


# ============================================================
# FILES
# ============================================================

ML_DIR = Path(__file__).resolve().parent

GSI_FILE = ML_DIR / "gsi_india_cleaned.csv"
NASA_FILE = ML_DIR / "india_landslides_cleaned.csv"

MASTER_FILE = ML_DIR / "india_master_landslide_inventory.csv"
NER_FILE = ML_DIR / "ner_master_landslide_inventory.csv"
MATCH_FILE = ML_DIR / "nasa_gsi_match_report.csv"


# ============================================================
# SETTINGS
# ============================================================

DUPLICATE_DISTANCE_KM = 1.0
YEAR_TOLERANCE = 2

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .replace("\n", " ")
    ).strip()


def extract_year(value):
    if pd.isna(value):
        return None

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        str(value)
    )

    if not years:
        return None

    return int(years[0])


def is_near_duplicate(
    nasa_year,
    gsi_year
):
    """
    Compare known years.

    If either year is unavailable, allow spatial matching
    to decide conservatively.
    """

    if nasa_year is None or gsi_year is None:
        return True

    return abs(
        nasa_year - gsi_year
    ) <= YEAR_TOLERANCE


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAKSHANET - INDIA LANDSLIDE SOURCE COMBINATION")
    print("==============================================")

    if not GSI_FILE.exists():
        raise FileNotFoundError(
            f"GSI file not found:\n{GSI_FILE}"
        )

    if not NASA_FILE.exists():
        raise FileNotFoundError(
            f"NASA file not found:\n{NASA_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    gsi = pd.read_csv(GSI_FILE)
    nasa = pd.read_csv(NASA_FILE)

    print(
        f"\nGSI records : {len(gsi):,}"
    )

    print(
        f"NASA records: {len(nasa):,}"
    )

    # ========================================================
    # CLEAN GSI
    # ========================================================

    gsi["latitude"] = pd.to_numeric(
        gsi["latitude"],
        errors="coerce"
    )

    gsi["longitude"] = pd.to_numeric(
        gsi["longitude"],
        errors="coerce"
    )

    gsi = gsi.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    gsi["source"] = "GSI"

    gsi["event_year"] = (
        gsi["history"]
        .apply(extract_year)
    )

    # ========================================================
    # CLEAN NASA
    # ========================================================

    nasa["latitude"] = pd.to_numeric(
        nasa["latitude"],
        errors="coerce"
    )

    nasa["longitude"] = pd.to_numeric(
        nasa["longitude"],
        errors="coerce"
    )

    nasa = nasa.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    nasa["source"] = "NASA"

    nasa["event_year"] = pd.to_numeric(
        nasa["event_year"],
        errors="coerce"
    )

    # ========================================================
    # PREPARE GSI COORDINATES
    # ========================================================

    gsi_coords = np.radians(
        gsi[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    nasa_coords = np.radians(
        nasa[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    # ========================================================
    # BALL TREE
    # ========================================================

    print(
        "\nBuilding spatial index..."
    )

    tree = BallTree(
        gsi_coords,
        metric="haversine"
    )

    radius_rad = (
        DUPLICATE_DISTANCE_KM
        / EARTH_RADIUS_KM
    )

    # ========================================================
    # FIND LIKELY NASA/GSI DUPLICATES
    # ========================================================

    print(
        "\nChecking NASA records against GSI..."
    )

    match_rows = []
    accepted_nasa_indexes = []

    duplicate_count = 0

    for i, row in nasa.iterrows():

        point = nasa_coords[
            i
        ].reshape(1, -1)

        indices, distances = tree.query_radius(
            point,
            r=radius_rad,
            return_distance=True
        )

        candidates = indices[0]
        candidate_distances = distances[0]

        best_match = None

        for gsi_index, distance_rad in zip(
            candidates,
            candidate_distances
        ):

            gsi_row = gsi.iloc[
                gsi_index
            ]

            distance_km = (
                distance_rad
                * EARTH_RADIUS_KM
            )

            gsi_year = (
                gsi_row["event_year"]
                if pd.notna(
                    gsi_row["event_year"]
                )
                else None
            )

            nasa_year = (
                row["event_year"]
                if pd.notna(
                    row["event_year"]
                )
                else None
            )

            if is_near_duplicate(
                nasa_year,
                gsi_year
            ):

                if (
                    best_match is None
                    or distance_km
                    < best_match["distance_km"]
                ):

                    best_match = {
                        "gsi_index": int(
                            gsi_index
                        ),
                        "distance_km":
                            distance_km,
                        "gsi_year":
                            gsi_year,
                        "nasa_year":
                            nasa_year,
                    }

        if best_match is not None:

            duplicate_count += 1

            gsi_row = gsi.iloc[
                best_match["gsi_index"]
            ]

            match_rows.append(
                {
                    "nasa_event_id":
                        row["event_id"],

                    "gsi_sl_no":
                        gsi_row["sl_no"],

                    "distance_km":
                        round(
                            best_match[
                                "distance_km"
                            ],
                            4
                        ),

                    "nasa_year":
                        best_match[
                            "nasa_year"
                        ],

                    "gsi_history_year":
                        best_match[
                            "gsi_year"
                        ],

                    "match_type":
                        "likely_duplicate",
                }
            )

        else:

            accepted_nasa_indexes.append(
                i
            )

            match_rows.append(
                {
                    "nasa_event_id":
                        row["event_id"],

                    "gsi_sl_no":
                        "",

                    "distance_km":
                        "",

                    "nasa_year":
                        row["event_year"],

                    "gsi_history_year":
                        "",

                    "match_type":
                        "additional_nasa_event",
                }
            )

    # ========================================================
    # NASA ADDITIONS
    # ========================================================

    nasa_additions = nasa.loc[
        accepted_nasa_indexes
    ].copy()

    print(
        f"\nLikely NASA/GSI duplicates : "
        f"{duplicate_count:,}"
    )

    print(
        f"Additional NASA events     : "
        f"{len(nasa_additions):,}"
    )

    # ========================================================
    # STANDARDIZE NASA INTO MASTER FORMAT
    # ========================================================

    nasa_master = pd.DataFrame(
        {
            "record_id": [
                f"NASA-{x}"
                for x in nasa_additions[
                    "event_id"
                ]
            ],

            "source":
                "NASA",

            "state":
                nasa_additions[
                    "state_normalized"
                ],

            "district":
                nasa_additions[
                    "admin_division_name"
                ],

            "slide_name":
                nasa_additions[
                    "event_title"
                ],

            "location":
                nasa_additions[
                    "location_description"
                ],

            "latitude":
                nasa_additions[
                    "latitude"
                ],

            "longitude":
                nasa_additions[
                    "longitude"
                ],

            "material":
                "",

            "movement_type":
                nasa_additions[
                    "landslide_category"
                ],

            "event_year":
                nasa_additions[
                    "event_year"
                ],

            "event_month":
                nasa_additions[
                    "event_month"
                ],

            "history":
                nasa_additions[
                    "event_date"
                ],

            "is_ner":
                nasa_additions[
                    "is_ner"
                ],
        }
    )

    # ========================================================
    # STANDARDIZE GSI
    # ========================================================

    gsi_master = pd.DataFrame(
        {
            "record_id": [
                f"GSI-{x}"
                for x in gsi[
                    "sl_no"
                ]
            ],

            "source":
                "GSI",

            "state":
                gsi[
                    "state_normalized"
                ],

            "district":
                gsi[
                    "district"
                ],

            "slide_name":
                gsi[
                    "slide_name"
                ],

            "location":
                gsi[
                    "nh_sh_location"
                ],

            "latitude":
                gsi[
                    "latitude"
                ],

            "longitude":
                gsi[
                    "longitude"
                ],

            "material":
                gsi[
                    "material_involved"
                ],

            "movement_type":
                gsi[
                    "movement_type_normalized"
                ],

            "event_year":
                gsi[
                    "event_year"
                ],

            "event_month":
                np.nan,

            "history":
                gsi[
                    "history"
                ],

            "is_ner":
                gsi[
                    "is_ner"
                ],
        }
    )

    # ========================================================
    # COMBINE
    # ========================================================

    master = pd.concat(
        [
            gsi_master,
            nasa_master
        ],
        ignore_index=True
    )

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    for column in [
        "record_id",
        "source",
        "state",
        "district",
        "slide_name",
        "location",
        "material",
        "movement_type",
        "history",
    ]:

        master[column] = (
            master[column]
            .apply(clean_text)
        )

    # ========================================================
    # FINAL SORT
    # ========================================================

    master = master.sort_values(
        by=[
            "state",
            "district",
            "event_year",
            "record_id"
        ],
        na_position="last"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # NER
    # ========================================================

    ner = master[
        master["is_ner"] == True
    ].copy()

    # ========================================================
    # MATCH REPORT
    # ========================================================

    match_report = pd.DataFrame(
        match_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    master.to_csv(
        MASTER_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    ner.to_csv(
        NER_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    match_report.to_csv(
        MATCH_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n=============================================="
    )
    print(
        "COMBINATION COMPLETE"
    )
    print(
        "=============================================="
    )

    print(
        f"\nGSI primary records       : "
        f"{len(gsi_master):,}"
    )

    print(
        f"NASA additional records   : "
        f"{len(nasa_master):,}"
    )

    print(
        f"Master inventory records  : "
        f"{len(master):,}"
    )

    print(
        f"NER master records        : "
        f"{len(ner):,}"
    )

    print(
        "\nSources:"
    )

    print(
        master[
            "source"
        ].value_counts()
        .to_string()
    )

    print(
        "\nNER by state:"
    )

    print(
        ner[
            "state"
        ].value_counts()
        .to_string()
    )

    print(
        "\nCreated:"
    )

    print(
        f"  ✅ {MASTER_FILE.name}"
    )

    print(
        f"  ✅ {NER_FILE.name}"
    )

    print(
        f"  ✅ {MATCH_FILE.name}"
    )


if __name__ == "__main__":
    main()