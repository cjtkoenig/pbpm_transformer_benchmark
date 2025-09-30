#!/usr/bin/env python3
"""
CSV → XES converter for PGTNet

This utility converts a canonical CSV event log under data/raw into an XES file
that the PGTNet converter expects. It relies on pm4py for dataframe→event log
conversion and XES export.

Default column mapping matches this repository's raw CSV expectations:
- case id:        case:concept:name
- activity name:  concept:name
- timestamp:      time:timestamp

Usage examples:
  # Convert Helpdesk.csv to data/raw/HelpDesk.xes
  uv run python scripts/csv_to_xes.py --dataset Helpdesk

  # Convert an arbitrary CSV to a specific .xes path
  uv run python scripts/csv_to_xes.py --csv /abs/path/MyLog.csv --out /abs/path/MyLog.xes

Notes:
- If pm4py is not installed in your current environment, either install it
  (pip install pm4py) or run the script using the helper env we provision for
  PGTNet/GraphGPS: third_party/graphgps_venv/bin/python scripts/csv_to_xes.py ...
- The tool will not modify your CSV. It writes a new .xes file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def fail(msg: str, code: int = 2) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert event log CSV to XES using pm4py.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=str, help="Path to input CSV file.")
    g.add_argument("--dataset", type=str, help="Dataset name; uses data/raw/<Dataset>.csv as input and data/raw/<Dataset>.xes as output by default.")

    parser.add_argument("--out", type=str, default=None, help="Path to output .xes (default: data/raw/<Dataset>.xes or alongside input CSV with .xes extension)")
    parser.add_argument("--pgtnet-profile", type=str, choices=["helpdesk", "trafficfines", "sepsis", "tourism"], default=None,
                        help="Apply PGTNet dataset-specific renaming so attribute names in XES match conversion_configs. Also sets default output filename to the YAML dataset_name unless --out is specified.")
    parser.add_argument("--case-column", type=str, default="case:concept:name", help="Case identifier column name (default: case:concept:name)")
    parser.add_argument("--activity-column", type=str, default="concept:name", help="Activity name column (default: concept:name)")
    parser.add_argument("--timestamp-column", type=str, default="time:timestamp", help="Timestamp column (default: time:timestamp)")
    parser.add_argument("--sep", type=str, default=",", help="CSV delimiter (default: ,)")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV encoding (default: utf-8)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .xes output if present")

    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception:
        fail("pandas is not installed in this Python environment. Install pandas or run via third_party/graphgps_venv/bin/python.")

    try:
        # Import pm4py pieces lazily and with a clear error if missing
        from pm4py.objects.log.util import dataframe_utils
        from pm4py.objects.conversion.log import converter as log_converter
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter
    except Exception as e:
        fail(
            "pm4py is not installed in this Python environment. Install it with `pip install pm4py` "
            "or run this script using the helper env: third_party/graphgps_venv/bin/python scripts/csv_to_xes.py ...\n"
            f"Original import error: {e}"
        )

    # Known PGTNet profiles from conversion_configs
    profiles = {
        "helpdesk": {
            "dataset_name": "HelpDesk.xes",
            "event_attributes": ['org:resource', 'workgroup', 'seriousness_2', 'service_level', 'service_type', 'customer'],
            "event_num_att": [],
            "case_attributes": ['responsible_section', 'support_section', 'product'],
            "case_num_att": [],
        },
        "trafficfines": {
            "dataset_name": "Traffic_Fines.xes",
            "event_attributes": ['org:resource', 'dismissal', 'vehicleClass', 'article', 'notificationType', 'lastSent'],
            "event_num_att": ['amount', 'totalPaymentAmount', 'points', 'expense', 'paymentAmount'],
            "case_attributes": [],
            "case_num_att": [],
        },
        "sepsis": {
            "dataset_name": "Sepsis.xes",
            "event_attributes": ['InfectionSuspected', 'org:group', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Diagnose', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG'],
            "event_num_att": ['Age', 'Leucocytes', 'CRP', 'LacticAcid'],
            "case_attributes": [],
            "case_num_att": [],
        },
        "tourism": {
            "dataset_name": "Tourism.xes",
            "event_attributes": ['org:resource'],  # minimal default; extend if needed
            "event_num_att": [],
            "case_attributes": [],
            "case_num_att": [],
        },
    }

    if args.dataset:
        in_csv = Path("data") / "raw" / f"{args.dataset}.csv"
        # Default output: dataset.xes next to CSV, unless a profile enforces specific filename
        if args.out:
            out_xes = Path(args.out)
        else:
            if args.pgtnet_profile and args.pgtnet_profile in profiles:
                out_xes = Path("data") / "raw" / profiles[args.pgtnet_profile]["dataset_name"]
            else:
                out_xes = Path("data") / "raw" / f"{args.dataset}.xes"
    else:
        in_csv = Path(args.csv)
        if args.out:
            out_xes = Path(args.out)
        else:
            if args.pgtnet_profile and args.pgtnet_profile in profiles:
                out_xes = in_csv.with_name(profiles[args.pgtnet_profile]["dataset_name"])  # same dir, profile filename
            else:
                out_xes = in_csv.with_suffix(".xes")

    if not in_csv.exists():
        fail(f"Input CSV not found: {in_csv}")

    if out_xes.exists() and not args.overwrite:
        fail(f"Output already exists: {out_xes}. Use --overwrite to replace.")

    # Read CSV
    try:
        df = pd.read_csv(in_csv, sep=args.sep, encoding=args.encoding)
    except Exception as e:
        fail(f"Failed to read CSV: {in_csv}\n{e}")

    # Validate columns with auto-detection of common aliases
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("_", "").replace(":", "")

    def _norm_case_attr_name(s: str) -> str:
        # normalize an attribute key disregarding optional 'case:' prefix
        s = str(s)
        if s.lower().startswith("case:"):
            s = s[5:]
        return _norm(s)

    provided = {
        "case": args.case_column,
        "activity": args.activity_column,
        "timestamp": args.timestamp_column,
    }
    aliases = {
        "case": ["case:concept:name", "case id", "caseid", "case", "case_concept_name", "event_log_case_id"],
        "activity": ["concept:name", "activity", "event", "eventname", "event_log_activity"],
        "timestamp": ["time:timestamp", "timestamp", "time", "complete timestamp", "end timestamp", "event time", "eventtime", "event_log_timestamp"],
    }

    # Build normalized lookup for DataFrame columns
    norm_to_col = { _norm(c): c for c in df.columns }

    chosen = {}
    inferred = {}
    for key in ("case", "activity", "timestamp"):
        col = provided[key]
        if col in df.columns:
            chosen[key] = col
            continue
        # Try aliases
        found = None
        for cand in aliases[key]:
            nc = _norm(cand)
            if nc in norm_to_col:
                found = norm_to_col[nc]
                break
        if found is None:
            # Try direct normalized match of the provided name
            nc = _norm(col)
            if nc in norm_to_col:
                found = norm_to_col[nc]
        if found is not None:
            chosen[key] = found
            inferred[key] = (col, found)
        else:
            fail(
                f"Required column '{col}' not found in CSV and no common alias matched. Available columns: {list(df.columns)}\n"
                "You can override names via --case-column/--activity-column/--timestamp-column."
            )

    if inferred:
        msg = ["Auto-detected columns:"]
        for k, (want, got) in inferred.items():
            msg.append(f"  {k}: '{want}' → '{got}'")
        print("\n".join(msg), file=sys.stderr)

    case_col = chosen["case"]
    act_col = chosen["activity"]
    ts_col = chosen["timestamp"]

    # Apply PGTNet profile-driven renaming for event/case attributes so XES keys match conversion YAMLs
    applied_mappings = []
    missing_targets = []
    if args.pgtnet_profile and args.pgtnet_profile in profiles:
        prof = profiles[args.pgtnet_profile]
        norm_to_col = { _norm(c): c for c in df.columns }
        # Event attributes (categorical + numeric)
        for t in list(prof.get("event_attributes", [])) + list(prof.get("event_num_att", [])):
            nt = _norm(t)
            # Prefer exact normalized match; also try some loose aliases
            candidates = [nt]
            if t == 'org:resource':
                candidates += [_norm('resource'), _norm('org_resource'), _norm('orgresource')]
            # Find first matching source column by normalized name among candidates
            src = None
            for cand in candidates:
                if cand in norm_to_col:
                    src = norm_to_col[cand]
                    break
            if src is not None:
                if src != t:
                    df = df.rename(columns={src: t})
                    applied_mappings.append((src, t))
            else:
                # Not found; record missing
                missing_targets.append(t)
        # Case attributes: ensure 'case:<name>' keys
        for t in list(prof.get("case_attributes", [])) + list(prof.get("case_num_att", [])):
            # Look for either 't' or 'case:t' variants
            nt = _norm_case_attr_name(t)
            src = None
            # Try exact with and without case: prefix in existing columns
            for c in df.columns:
                if _norm_case_attr_name(c) == nt:
                    src = c
                    break
            target_col = f"case:{t}"  # enforce case: prefix
            if src is not None:
                if src != target_col:
                    df = df.rename(columns={src: target_col})
                    applied_mappings.append((src, target_col))
            else:
                missing_targets.append(target_col)
        if applied_mappings:
            print("Applied PGTNet profile column mappings:", file=sys.stderr)
            for s, d in applied_mappings:
                print(f"  {s} → {d}", file=sys.stderr)
        if missing_targets:
            print("Warning: the following PGTNet-expected attributes were not found in the CSV:", file=sys.stderr)
            for m in missing_targets:
                print(f"  - {m}", file=sys.stderr)

    # Normalize timestamps and ensure correct dtypes
    try:
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        # pm4py's helper converts any column named like a timestamp to pandas datetime; ensure our key exists
        if df[ts_col].dtype.kind != 'M':  # not datetime64
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        if df[ts_col].isna().any():
            # Drop rows without valid timestamps to avoid exporter failures
            before = len(df)
            df = df.dropna(subset=[ts_col])
            after = len(df)
            if after < before:
                print(f"Warning: dropped {before-after} rows with invalid timestamps", file=sys.stderr)
    except Exception as e:
        fail(f"Failed to normalize timestamps: {e}")

    # Prepare a copy with canonical pm4py column names to avoid version-specific parameter issues
    df_conv = df.rename(columns={
        case_col: "case:concept:name",
        act_col: "concept:name",
        ts_col: "time:timestamp",
    })

    # HARD-CODED HELPDesK MAPPING (as requested):
    # If this looks like the Helpdesk dataset (by name or by columns), enforce the exact XES keys:
    # Trace-level: case:concept:name (Case ID), case:responsible_section, case:support_section, case:product
    # Event-level: concept:name (Activity), time:timestamp (ISO-8601), org:resource (from Resource),
    #              workgroup, seriousness_2, service_level, service_type, customer
    def _looks_like_helpdesk() -> bool:
        cols = set(c.lower() for c in df.columns)
        required = {"case id", "activity", "resource", "complete timestamp"}
        return (args.dataset and args.dataset.lower() == "helpdesk") or required.issubset(cols)

    if _looks_like_helpdesk():
        import pandas as _pd
        # 1) Rename Resource -> org:resource for events
        if "Resource" in df_conv.columns:
            df_conv = df_conv.rename(columns={"Resource": "org:resource"})
        if "resource" in df_conv.columns:
            df_conv = df_conv.rename(columns={"resource": "org:resource"})
        # 2) Promote case attributes to case: prefixed trace attributes using first non-null value per case
        case_id_col = "case:concept:name"
        _case_attrs = [
            ("responsible_section", "case:responsible_section"),
            ("support_section", "case:support_section"),
            ("product", "case:product"),
        ]
        # Ensure the source columns exist (case-insensitive)
        norm_map = {str(c).lower(): c for c in df_conv.columns}
        for src_name, tgt_name in _case_attrs:
            src_lookup = src_name
            if src_lookup not in norm_map:
                # try exact as-is from CSV
                if src_name in df_conv.columns:
                    src_lookup = src_name
                else:
                    # nothing to do if missing
                    continue
            src_col = norm_map.get(src_lookup, src_lookup)
            # Compute first non-null value per case and assign as a case attribute column
            if src_col in df_conv.columns:
                # Create per-case attribute column by taking first non-null per case
                first_vals = (
                    df_conv[[case_id_col, src_col]]
                    .dropna(subset=[src_col])
                    .groupby(case_id_col)[src_col]
                    .first()
                )
                # Map back to all rows (pm4py will convert 'case:' columns to trace attrs)
                df_conv[tgt_name] = df_conv[case_id_col].map(first_vals)
                # Remove the event-level column to avoid duplicating at event level
                if src_col in df_conv.columns:
                    df_conv = df_conv.drop(columns=[src_col])
        # 3) Filter to keep ONLY the required event and trace attributes
        allowed_event_attrs = [
            "org:resource",
            "workgroup",
            "seriousness_2",
            "service_level",
            "service_type",
            "customer",
        ]
        allowed_trace_attrs = ["case:responsible_section", "case:support_section", "case:product"]
        base_cols = ["case:concept:name", "concept:name", "time:timestamp"]
        keep_cols = [c for c in base_cols + allowed_event_attrs + allowed_trace_attrs if c in df_conv.columns]
        df_conv = df_conv[keep_cols]
        # 4) Ensure timestamps are timezone-aware ISO-8601 (UTC)
        if _pd.api.types.is_datetime64_any_dtype(df_conv["time:timestamp"]) and df_conv["time:timestamp"].dt.tz is None:
            df_conv["time:timestamp"] = df_conv["time:timestamp"].dt.tz_localize("UTC")

    # Convert to event log and export to XES
    try:
        event_log = log_converter.apply(df_conv, variant=log_converter.Variants.TO_EVENT_LOG)
    except Exception as e:
        fail(f"Failed to convert dataframe to event log: {e}")

    try:
        out_xes.parent.mkdir(parents=True, exist_ok=True)
        xes_exporter.apply(event_log, str(out_xes))
    except Exception as e:
        fail(f"Failed to export XES to {out_xes}: {e}")

    print(f"Wrote XES: {out_xes}")
    # Helpful hint for PGTNet
    print(
        "Hint: For PGTNet, place the XES under third_party/PGTNet/raw_dataset with the exact name expected by "
        "conversion_configs/<cfg>.yaml (dataset_name), or pass +model.pgtnet.converter.input_xes=/abs/path/file.xes\n"
        "Alternatively, you can keep it in data/raw and the runner will try to copy it automatically."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
