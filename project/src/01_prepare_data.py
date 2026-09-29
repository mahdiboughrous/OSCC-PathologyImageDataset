#!/usr/bin/env python3
"""
01_prepare_data.py - Merge, verify, and clean the OSCC dataset.
"""

import json
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLINICAL_FILE = DATA_DIR / "clinical_data_2024.csv"
METADATA_FILE = DATA_DIR / "all_metadata.json"
SPLIT_FILE = DATA_DIR / "split_seed=2024.json"
OUTPUT_CSV = DATA_DIR / "processed_data.csv"
OUTPUT_SUMMARY = DATA_DIR / "prepared_summary.txt"

# Column lists
PRE_TREATMENT_FEATURES = [
    "Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)",
    "AlcoholHistory(0no/1yes)", "SmokingHistory(0no/1yes)",
    "BetelNutHistory(0no/1yes)", "ASAGrade", "Surgery Site",
    "HPV(0/1)", "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
    "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)",
]
PATHOLOGY_LABELS = ["TD", "TI", "CE", "PI"]
TARGET = "REC"
DROP_COLUMNS = [
    "SurgicalMethod", "NNISGrade", "SurgicalMargin(0/1)", "6-Class Flap",
    "NeckDissection", "Tracheotomy(0no/1yes)", "Radiotherapy(0no/1yes)",
    "Chemotherapy(0no/1yes)", "[annotation] recurrence time",
    "[annotation] last followup time", "path",
]

print("Loading raw data...")
clin_df = pd.read_csv(CLINICAL_FILE)
clin_df["PID"] = pd.to_numeric(clin_df["PID"], errors="coerce").astype("Int64")
print(f"  Clinical: {len(clin_df)} rows")

with open(METADATA_FILE, "r") as f:
    meta_raw = json.load(f)
meta_df = pd.DataFrame(meta_raw["datainfo"])
meta_df["pid"] = pd.to_numeric(meta_df["pid"], errors="coerce").astype("Int64")
print(f"  Metadata: {len(meta_df)} rows")

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)
print(f"  Splits: {list(splits.keys())} = {[len(splits[k]) for k in splits.keys()]}")

print("\nMerging clinical x metadata on PID...")
merged = pd.merge(clin_df, meta_df, left_on="PID", right_on="pid", how="inner")
print(f"  Matched: {len(merged)} patients")
assert len(merged) == 1325, f"Expected 1325, got {len(merged)}"
assert merged["PID"].is_unique and merged["pid"].is_unique
print("  OK: 1325 unique patients confirmed.")

rec_vals = merged[TARGET].value_counts().sort_index()
print(f"\nREC distribution:")
for val, cnt in rec_vals.items():
    print(f"  REC={val}: {cnt} ({cnt/len(merged)*100:.2f}%)")
assert rec_vals.get(1, 0) == 275
assert rec_vals.get(0, 0) == 1050
print("  OK: REC distribution verified.")

print("\nSelecting features...")
use_cols = PRE_TREATMENT_FEATURES + PATHOLOGY_LABELS + [TARGET, "PID"]
keep = [c for c in use_cols if c in merged.columns]
out = merged[keep].copy()
for col in DROP_COLUMNS:
    if col in out.columns:
        out = out.drop(columns=[col])
print(f"  Final shape: {out.shape[0]} x {out.shape[1]}")

print("\nAssigning splits...")
split_map = {int(pid): split_name for split_name, pids in splits.items() for pid in pids}
out["split"] = out["PID"].map(split_map)
assert out["split"].notna().all()
print(f"  Split counts:\n{out['split'].value_counts().sort_index()}")

print("\nData quality checks:")
for col in out.columns:
    if col != "PID":
        non_null = out[col].notna().sum()
        pct = (1 - non_null / len(out)) * 100
        if pct > 0:
            print(f"  {col}: {non_null}/{len(out)} non-null ({pct:.1f}% missing)")

print(f"\nSaving to {OUTPUT_CSV}...")
out.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved {len(out)} rows.")

print(f"Writing summary to {OUTPUT_SUMMARY}...")
with open(OUTPUT_SUMMARY, "w") as f:
    f.write(f"Patients: {len(out)}\n")
    f.write(f"Columns: {out.shape[1]}\n\n")
    f.write("REC distribution:\n")
    for val, cnt in rec_vals.items():
        f.write(f"  REC={val}: {cnt} ({cnt/len(out)*100:.2f}%)\n\n")
    f.write("Split counts:\n")
    f.write(out["split"].value_counts().sort_index().to_string() + "\n\n")
    f.write("Feature columns:\n")
    for col in PRE_TREATMENT_FEATURES + PATHOLOGY_LABELS:
        non_null = out[col].notna().sum()
        f.write(f"  {col}: {non_null}/{len(out)} non-null\n")
print("  Done.")
print("\nPhase 1 complete. Proceed to Phase 2.")
