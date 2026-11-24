"""Batch file ingestion and schema validation for evaluation datasets.

Supports CSV, TSV, JSON, and XLSX formats with UTF-8 and Devanagari handling.
Strictly validates required columns, JSON-parsable fields, duplicates, and empty rows.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_CORE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
try:
    from core.eval.normalizer import normalize_unicode
    from core.eval.registry import REGISTRY, ComponentDef
except ModuleNotFoundError:
    from eval.normalizer import normalize_unicode
    from eval.registry import REGISTRY, ComponentDef


def parse_and_validate_batch(
    content_bytes: bytes,
    filename: str,
    component_id: str,
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse upload bytes and validate against component schema.
    
    Returns:
        (is_valid, parsed_rows, row_errors)
    """
    comp_def: ComponentDef = REGISTRY.get(component_id)
    if not comp_def:
        return False, [], [{"row": 0, "error": f"Unknown component: {component_id}"}]

    # 1. Parse into DataFrame
    ext = Path(filename).suffix.lower()
    df: pd.DataFrame
    try:
        if ext in (".csv", ".txt"):
            # Handles UTF-8 with or without BOM
            df = pd.read_csv(io.BytesIO(content_bytes), encoding="utf-8-sig", dtype=str)
        elif ext == ".tsv":
            df = pd.read_csv(io.BytesIO(content_bytes), sep="\t", encoding="utf-8-sig", dtype=str)
        elif ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(content_bytes), dtype=str)
        elif ext == ".json":
            raw_text = content_bytes.decode("utf-8-sig")
            data = json.loads(raw_text)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "rows" in data:
                df = pd.DataFrame(data["rows"])
            elif isinstance(data, dict) and "items" in data:
                df = pd.DataFrame(data["items"])
            else:
                return False, [], [{"row": 0, "error": "JSON must be a list of records or contain 'rows' key"}]
        else:
            return False, [], [{"row": 0, "error": f"Unsupported file extension: {ext}. Use .csv, .xlsx, or .json"}]
    except Exception as e:
        return False, [], [{"row": 0, "error": f"Failed to parse file: {str(e)}"}]

    if df.empty:
        return False, [], [{"row": 0, "error": "The uploaded file is empty"}]

    # 2. Normalize column headers
    col_map = {col: str(col).strip().lower() for col in df.columns}
    df.rename(columns=col_map, inplace=True)

    # 3. Check required columns
    required_cols = [c.name for c in comp_def.columns if c.required]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, [], [{"row": 0, "error": f"Missing required columns: {', '.join(missing_cols)}"}]

    # 4. Determine ID column
    id_col = "query_id" if "query_id" in df.columns else ("doc_id" if "doc_id" in df.columns else "row_id")

    # 5. Row-by-row validation
    validated_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, raw_row in df.iterrows():
        row_num = idx + 2  # 1-based + 1 for header
        row_dict = raw_row.dropna().to_dict()

        # Check empty row
        if not any(str(v).strip() for v in row_dict.values()):
            errors.append({"row": row_num, "error": "Empty row skipped"})
            continue

        # Extract and validate ID
        row_id_val = str(row_dict.get(id_col, f"row_{row_num}")).strip()
        if not row_id_val or row_id_val == "nan":
            errors.append({"row": row_num, "error": f"Missing value for ID column '{id_col}'"})
            continue
        if row_id_val in seen_ids:
            errors.append({"row": row_num, "error": f"Duplicate ID '{row_id_val}'"})
            continue
        seen_ids.add(row_id_val)

        # Validate typed fields
        cleaned_row: Dict[str, Any] = {"_row_num": row_num, id_col: row_id_val}
        row_has_err = False

        for col_def in comp_def.columns:
            val = row_dict.get(col_def.name)
            if val is None or pd.isna(val) or str(val).strip() == "":
                if col_def.required:
                    errors.append({"row": row_num, "error": f"Missing required field '{col_def.name}'"})
                    row_has_err = True
                    break
                else:
                    cleaned_row[col_def.name] = None
                    continue

            val_str = str(val).strip()
            # Unicode NFC normalization for all string fields
            val_str = normalize_unicode(val_str)

            if col_def.col_type == "json_list":
                parsed_list = _parse_list_field(val_str)
                if parsed_list is None:
                    errors.append({"row": row_num, "error": f"Field '{col_def.name}' must be a JSON list or comma-separated string (got: {val_str[:40]})"})
                    row_has_err = True
                    break
                cleaned_row[col_def.name] = parsed_list

            elif col_def.col_type == "json_dict":
                try:
                    cleaned_row[col_def.name] = json.loads(val_str)
                except Exception:
                    errors.append({"row": row_num, "error": f"Field '{col_def.name}' must be valid JSON object (got: {val_str[:40]})"})
                    row_has_err = True
                    break

            elif col_def.col_type == "bool":
                low = val_str.lower()
                if low in ("true", "1", "yes", "t", "y"):
                    cleaned_row[col_def.name] = True
                elif low in ("false", "0", "no", "f", "n"):
                    cleaned_row[col_def.name] = False
                else:
                    errors.append({"row": row_num, "error": f"Field '{col_def.name}' must be a boolean (got: {val_str})"})
                    row_has_err = True
                    break

            elif col_def.col_type == "int":
                try:
                    cleaned_row[col_def.name] = int(float(val_str))
                except ValueError:
                    errors.append({"row": row_num, "error": f"Field '{col_def.name}' must be an integer (got: {val_str})"})
                    row_has_err = True
                    break
            else:
                cleaned_row[col_def.name] = val_str

        if not row_has_err:
            validated_rows.append(cleaned_row)

    is_valid = len(validated_rows) > 0 and len(errors) == 0
    return is_valid, validated_rows, errors


def _parse_list_field(raw: str) -> Optional[List[str]]:
    """Parse a field that can be either a JSON array `["a", "b"]` or comma/newline separated `a, b`."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    # Fallback to comma or newline separated
    parts = [p.strip() for p in re.split(r"[,;\n]+", raw) if p.strip()]
    return parts if parts else None

