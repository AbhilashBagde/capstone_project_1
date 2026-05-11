"""
build_sst.py
Transforms raw SST data into the canonical SST_v6 format.

Inputs (set paths below or pass as CLI args):
  --sst   Path to the raw SST CSV (national or 5-state, must include SDOH columns)
  --road  Path to the road-distance CSV (CCN, Year, nearest_building_road_dist_miles)
  --out   Output path (default: SST_v6.csv)

The output is validated against the exact SST_v6 column schema before saving.
Run with --validate-only to check an existing file without regenerating.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Canonical column schema (exact order from SST_v6.csv) ─────────────────────

SST_V6_COLUMNS = [
    'Unnamed: 0',
    'CCN',
    'Hospital_Name',
    'State',
    'City',
    'Zip',
    'County',
    'Address',
    'Year',
    'FY_Begin_NASHP',
    'FY_End_NASHP',
    'FY_Begin_CMS',
    'FY_End_CMS',
    'Data_Source',
    'Data_Source_CMS',
    'Data_Source_NASHP',
    'Facility_Type',
    'Facility_Type_CMS',
    'Facility_Type_NASHP',
    'Rural_Urban',
    'RUCA_Code',
    'CBSA',
    'Is_CAH',
    'Is_REH_Converter',
    'REH_Conversion_Date',
    'Post_REH_Months',
    'REH_Status',
    'Pre_REH_CCN',
    'Pre_REH_Payment_Type',
    'Ownership_Type',
    'Health_System',
    'Health_System_ID',
    'Is_Independent',
    'Payment_Type',
    'Type_of_Control',
    'Num_Beds',
    'NASHP_Bed_Size',
    'FTE_Employees',
    'Total_Discharges',
    'Total_Patient_Days',
    'Medicare_Days',
    'Medicaid_Days',
    'Medicare_Discharges',
    'Medicaid_Discharges',
    'Adjusted_Patient_Discharges',
    'Inpatient_Occupancy',
    'Operating_Margin_Pct',
    'Net_Profit_Margin_Pct',
    'Current_Ratio',
    'Debt_Ratio',
    'Medicare_Pct_Days',
    'Medicaid_Pct_Days',
    'Outpatient_Rev_Pct',
    'Days_Cash_on_Hand',
    'Labor_Cost_Pct',
    'Rev_per_Discharge',
    'Cost_per_Discharge',
    'Rev_per_Adj_Discharge',
    'Cost_per_Adj_Discharge',
    'Charity_Care_Pct_Revenue',
    'CMS_Inpatient_Revenue',
    'CMS_Outpatient_Revenue',
    'CMS_Net_Patient_Revenue',
    'CMS_Total_Costs',
    'CMS_Operating_Expense',
    'CMS_Net_Income_from_Patients',
    'CMS_Other_Income',
    'CMS_Net_Income',
    'CMS_Total_Salaries',
    'CMS_Charity_Care_Cost',
    'CMS_Bad_Debt_Expense',
    'CMS_Uncompensated_Care',
    'CMS_Total_Assets',
    'CMS_Total_Liabilities',
    'CMS_Current_Assets',
    'CMS_Current_Liabilities',
    'CMS_Cash',
    'CMS_Depreciation',
    'CMS_DSH_Payment',
    'CMS_DSH_Pct',
    'CMS_Medicaid_Revenue',
    'Cost_to_Charge_Ratio',
    'Fund_Balance',
    'NASHP_Net_Patient_Revenue',
    'NASHP_Operating_Expenses',
    'NASHP_Net_Income',
    'NASHP_Net_Profit_Margin',
    'NASHP_Hospital_Op_Costs',
    'NASHP_Charity_Care_Cost',
    'NASHP_Uninsured_Bad_Debt',
    'NASHP_Operating_Profit',
    'NASHP_Operating_Margin',
    'NASHP_Geo_Classification',
    'Total_Hospital_Expenses',
    'Medicaid_Payer_Mix_Pct',
    'SCHIP_Payer_Mix_Pct',
    'Medicare_Payer_Mix_Pct',
    'Medicare_Adv_Payer_Mix_Pct',
    'Commercial_Payer_Mix_Pct',
    'Medicaid_Revenue_Pct',
    'Medicaid_Op_Margin',
    'Medicare_Op_Margin',
    'Medicare_Adv_Op_Margin',
    'Commercial_Op_Margin',
    'COVID_PHE_Funding',
    'Direct_Labor_Cost',
    'Contract_Labor_Cost',
    'Hospital_Name_CMS',
    'Hospital_Name_NASHP',
    'Provider_Type',
    'NASHP_Fund_Balance',
    'Is_340B_Enrolled',
    '_county',
    'StateAbbr',
    'LocationName',
    'Any disability among adults',
    'Cancer (non-skin) or melanoma among adults',
    'Current lack of health insurance among adults aged 18-64 years',
    'Fair or poor self-rated health status among adults',
    'Food insecurity in the past 12 months among adults',
    'Obesity among adults',
    'Cancer_PCTL',
    'Obesity_PCTL',
    'Food_Insecurity_PCTL',
    'Uninsured_PCTL',
    'Disability_PCTL',
    'Poor_Health_PCTL',
    'nearest_building_road_dist_miles',
]

FIVE_STATES = ['AR', 'LA', 'NM', 'OK', 'TX']


def validate_schema(df: pd.DataFrame, label: str = '') -> list[str]:
    """Return list of schema error strings; empty = valid."""
    errors = []
    actual = list(df.columns)
    expected = SST_V6_COLUMNS

    missing  = [c for c in expected if c not in actual]
    extra    = [c for c in actual   if c not in expected]
    if missing:
        errors.append(f"Missing columns ({len(missing)}): {missing}")
    if extra:
        errors.append(f"Unexpected extra columns ({len(extra)}): {extra}")

    # Check ordering only when no missing/extra columns
    if not missing and not extra and actual != expected:
        first_wrong = next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)
        errors.append(
            f"Column order mismatch starting at position {first_wrong}: "
            f"got '{actual[first_wrong]}', expected '{expected[first_wrong]}'"
        )

    state_vals = df['State'].unique() if 'State' in df.columns else []
    bad_states = [s for s in state_vals if s not in FIVE_STATES]
    if bad_states:
        errors.append(f"Rows with out-of-scope states found: {bad_states}")

    tag = f"[{label}] " if label else ""
    return [f"{tag}{e}" for e in errors]


def build(sst_path: str, road_path: str, out_path: str) -> pd.DataFrame:
    """Full build pipeline → returns the final DataFrame."""

    # ── 1. Load SST data ──────────────────────────────────────────────────────
    print(f"Loading SST data from: {sst_path}")
    df = pd.read_csv(sst_path, low_memory=False)
    print(f"  {len(df):,} rows, {df['CCN'].nunique():,} unique CCNs loaded")

    # ── 2. Filter to 5-state region ───────────────────────────────────────────
    df = df[df['State'].isin(FIVE_STATES)].copy()
    print(f"  After 5-state filter: {len(df):,} rows, {df['CCN'].nunique():,} CCNs")

    # ── 3. Reset the row-level index (Unnamed: 0) ─────────────────────────────
    # Unnamed: 0 is a sequential integer index preserved from the upstream pipeline.
    # We reset it so it is contiguous for this 5-state slice.
    df = df.reset_index(drop=True)
    df.insert(0, 'Unnamed: 0', df.index)

    # ── 4. Load and deduplicate road distance ─────────────────────────────────
    print(f"Loading road distance data from: {road_path}")
    road = pd.read_csv(road_path, dtype={'CCN': str, 'Year': int})
    road['CCN'] = road['CCN'].astype(str).str.zfill(6)

    # Keep the most recent year's value per CCN
    road_deduped = (
        road.sort_values('Year', ascending=False)
        .drop_duplicates(subset='CCN', keep='first')
        [['CCN', 'nearest_building_road_dist_miles']]
    )
    print(f"  {len(road_deduped):,} unique CCNs with road distance")

    # ── 5. Merge road distance ────────────────────────────────────────────────
    df['CCN'] = df['CCN'].astype(str).str.zfill(6)

    # Drop stale road-distance column if it already exists in the SST file
    if 'nearest_building_road_dist_miles' in df.columns:
        df = df.drop(columns=['nearest_building_road_dist_miles'])

    df = df.merge(road_deduped, on='CCN', how='left')

    populated = df['nearest_building_road_dist_miles'].notna().sum()
    print(f"  Road distance populated: {populated:,} / {len(df):,} rows")

    # ── 6. Enforce exact column schema ────────────────────────────────────────
    missing_cols = [c for c in SST_V6_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"\n⚠️  Adding {len(missing_cols)} missing columns as NaN: {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan

    extra_cols = [c for c in df.columns if c not in SST_V6_COLUMNS]
    if extra_cols:
        print(f"\n⚠️  Dropping {len(extra_cols)} extra columns not in schema: {extra_cols}")
        df = df.drop(columns=extra_cols)

    df = df[SST_V6_COLUMNS]

    return df


def validate_file(path: str):
    """Load an existing CSV and validate its schema."""
    df = pd.read_csv(path, low_memory=False, nrows=0)  # columns only
    errors = validate_schema(df, label=path)
    if errors:
        print(f"SCHEMA ERRORS in {path}:")
        for e in errors:
            print(f"  ✗ {e}")
        return False
    else:
        print(f"✓ {path} matches SST_v6 schema ({len(SST_V6_COLUMNS)} columns)")
        return True


def main():
    parser = argparse.ArgumentParser(description='Build SST_v6-format dataset')
    parser.add_argument('--sst',  default='SST_raw.csv',
                        help='Path to raw SST CSV (with SDOH columns merged in)')
    parser.add_argument('--road', default='../updated_df_road_dist.csv',
                        help='Path to road-distance CSV')
    parser.add_argument('--out',  default='SST_v6.csv',
                        help='Output path for the final dataset')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate an existing output file without rebuilding')
    args = parser.parse_args()

    if args.validate_only:
        ok = validate_file(args.out)
        sys.exit(0 if ok else 1)

    # Check inputs exist
    for path, name in [(args.sst, '--sst'), (args.road, '--road')]:
        if not Path(path).exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    df = build(args.sst, args.road, args.out)

    # Validate before saving
    errors = validate_schema(df)
    if errors:
        print("\nSCHEMA VALIDATION FAILED — output not saved:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    df.to_csv(args.out, index=False)
    print(f"\n✓ Saved {args.out}")
    print(f"  Rows: {len(df):,} | CCNs: {df['CCN'].nunique():,} | Columns: {len(df.columns)}")
    print(f"  States: {sorted(df['State'].unique())}")
    print(f"  Years:  {sorted(df['Year'].unique())}")

    # Re-validate the saved file (round-trip check)
    saved_cols = list(pd.read_csv(args.out, nrows=0).columns)
    if saved_cols != SST_V6_COLUMNS:
        print("⚠️  Round-trip column check failed — saved file column order differs.")
    else:
        print("✓ Round-trip column validation passed.")


if __name__ == '__main__':
    main()
