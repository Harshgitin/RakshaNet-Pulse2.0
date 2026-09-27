import csv
import re
from pathlib import Path

import pdfplumber


# ============================================================
# FILES
# ============================================================

ML_DIR = Path(__file__).resolve().parent

# Put your compressed GSI PDF inside ml/
PDF_FILE = ML_DIR / "landslide_report_compressed.pdf"

OUTPUT_FILE = ML_DIR / "GSI_landslide_inventory.csv"


HEADERS = [
    "sl_no",
    "slide_no",
    "state",
    "district",
    "slide_name",
    "nh_sh_location",
    "latitude",
    "longitude",
    "material_involved",
    "movement_type",
    "history",
]


# ============================================================
# CLEAN CELL
# ============================================================

def clean_cell(value):
    if value is None:
        return ""

    value = str(value)

    # Join wrapped text into one line.
    value = value.replace("\n", " ")

    # Remove repeated spaces.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# MAIN EXTRACTION
# ============================================================

def main():

    print("\n==============================================")
    print("RAKSHANET - GSI INVENTORY EXTRACTION")
    print("==============================================")

    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"\nGSI PDF not found:\n{PDF_FILE}\n\n"
            "Put landslide_report_compressed.pdf inside the ml folder."
        )

    rows = []
    seen_sl_numbers = set()

    with pdfplumber.open(PDF_FILE) as pdf:

        total_pages = len(pdf.pages)

        print(f"\nPDF pages: {total_pages}")
        print("Starting table extraction...\n")

        for page_number, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                for raw_row in table:

                    if not raw_row:
                        continue

                    row = [
                        clean_cell(cell)
                        for cell in raw_row
                    ]

                    # We expect 11 columns.
                    if len(row) < 11:
                        row += [""] * (11 - len(row))

                    if len(row) > 11:
                        row = row[:11]

                    sl_no = row[0]

                    # Ignore headings/repeated headers.
                    if not sl_no.isdigit():
                        continue

                    # Avoid duplicate extraction.
                    if sl_no in seen_sl_numbers:
                        continue

                    seen_sl_numbers.add(sl_no)

                    rows.append(row)

            if page_number % 50 == 0 or page_number == total_pages:
                print(
                    f"Processed {page_number}/{total_pages} pages "
                    f"-> {len(rows):,} rows"
                )

    # ========================================================
    # SORT BY SERIAL NUMBER
    # ========================================================

    rows.sort(
        key=lambda x: int(x[0])
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(HEADERS)

        writer.writerows(rows)

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n==============================================")
    print("EXTRACTION COMPLETE")
    print("==============================================")

    print(
        f"\nExtracted records: {len(rows):,}"
    )

    print(
        f"Output file:\n{OUTPUT_FILE}"
    )

    # Expected GSI inventory is around 36k records.
    if 35000 <= len(rows) <= 37000:
        print(
            "\n✅ Record count looks consistent with the "
            "GSI field-validated inventory."
        )
    else:
        print(
            "\n⚠️ Record count is outside the expected range. "
            "We'll inspect it before using it for ML."
        )

    print("\nFirst 3 records:")

    for row in rows[:3]:
        print(row)

    print("\nLast 3 records:")

    for row in rows[-3:]:
        print(row)


if __name__ == "__main__":
    main()