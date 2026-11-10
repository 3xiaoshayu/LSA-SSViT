#!/usr/bin/env python
"""Build the INbreast manifest for the external test without fine-tuning.

Reads INbreast.xls and the DICOM folder of the official INbreast release and writes one
row per image (410 images, 108 patients). Labels:
    screening      BI-RADS first digit >= 3 (includes 4a/4b/4c and 6), as for VinDr-Mammo
    mass           "Mass" column of INbreast.xls (benign findings included)
    calcification  "Micros" column of INbreast.xls (benign findings included)
The patient identifier (second field of the DICOM file name) is used as study_id, so
that bootstrap resampling is done over patients. The script stops if the counts differ
from the published dataset. Reading .xls files requires the xlrd package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import parse_birads_ge3  # noqa: E402

EXPECTED_ROWS = 410
EXPECTED_PATIENTS = 108
EXPECTED_POSITIVE = {"birads": 123, "mass": 108, "calcification": 308}
EXPECTED_BIRADS6 = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the INbreast manifest.")
    parser.add_argument("--xls", type=Path, default=PROJECT_ROOT / "dataset" / "inbreast" / "INbreast.xls")
    parser.add_argument("--dicom-dir", type=Path, default=PROJECT_ROOT / "dataset" / "inbreast" / "AllDICOMs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inbreast" / "manifest.csv")
    return parser.parse_args()


def integer_text(value: object) -> str:
    """Excel stores numbers as floats; 20586908.0 -> '20586908'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def flag_bit(value: object) -> int:
    return int(str(value).strip().upper() == "X")


def dicom_size(path: Path) -> tuple[int, int]:
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
    return int(dataset.Rows), int(dataset.Columns)


def find_dicom(dicom_dir: Path, file_id: str) -> Path:
    matches = sorted(dicom_dir.glob(f"{file_id}_*.dcm"))
    if len(matches) != 1:
        raise RuntimeError(f"File name {file_id}: expected exactly one DICOM, found {len(matches)}")
    return matches[0]


def build(xls_path: Path, dicom_dir: Path) -> pd.DataFrame:
    sheet = pd.read_excel(xls_path, sheet_name="Sheet1")
    columns = {str(name).strip(): name for name in sheet.columns}
    needed = ["File Name", "Bi-Rads", "Laterality", "View", "Mass", "Micros"]
    missing = [name for name in needed if name not in columns]
    if missing:
        raise RuntimeError(f"{xls_path} is missing columns: {missing}")
    sheet = sheet[sheet[columns["File Name"]].notna()]
    rows = []
    for record in sheet.to_dict(orient="records"):
        file_id = integer_text(record[columns["File Name"]])
        path = find_dicom(dicom_dir, file_id)
        parts = path.stem.split("_")
        if len(parts) < 2 or parts[0] != file_id:
            raise RuntimeError(f"Unexpected DICOM file name: {path.name}")
        height, width = dicom_size(path)
        birads = integer_text(record[columns["Bi-Rads"]])
        parse_birads_ge3(birads)  # fails loudly on an unreadable category
        rows.append({
            "study_id": parts[1],
            "series_id": "inbreast",
            "image_id": file_id,
            "image_path": path.name,
            "laterality": str(record[columns["Laterality"]]).strip(),
            "view_position": str(record[columns["View"]]).strip(),
            "height": height,
            "width": width,
            "breast_birads": birads,
            "mass": flag_bit(record[columns["Mass"]]),
            "calcification": flag_bit(record[columns["Micros"]]),
        })
    return pd.DataFrame(rows)


def check(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_ROWS or frame["image_id"].duplicated().any():
        raise RuntimeError(f"Expected {EXPECTED_ROWS} unique images, found {len(frame)}")
    if int(frame["study_id"].nunique()) != EXPECTED_PATIENTS:
        raise RuntimeError(f"Expected {EXPECTED_PATIENTS} patients, found {frame['study_id'].nunique()}")
    counts = {
        "birads": int(frame["breast_birads"].map(parse_birads_ge3).sum()),
        "mass": int(frame["mass"].sum()),
        "calcification": int(frame["calcification"].sum()),
    }
    if counts != EXPECTED_POSITIVE:
        raise RuntimeError(f"Positive counts {counts} differ from {EXPECTED_POSITIVE}")
    birads6 = int((frame["breast_birads"].astype(str) == "6").sum())
    if birads6 != EXPECTED_BIRADS6:
        raise RuntimeError(f"Expected {EXPECTED_BIRADS6} BI-RADS 6 images, found {birads6}")


def main() -> None:
    args = parse_args()
    if not args.xls.is_file() or not args.dicom_dir.is_dir():
        raise FileNotFoundError(f"Need {args.xls} and the DICOM folder {args.dicom_dir}")
    frame = build(args.xls, args.dicom_dir)
    check(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(frame)} images, {frame['study_id'].nunique()} patients, "
          f"screening-positive {int(frame['breast_birads'].map(parse_birads_ge3).sum())}")


if __name__ == "__main__":
    main()
