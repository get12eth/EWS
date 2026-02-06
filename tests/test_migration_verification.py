"""Integration test: migration verification

This test compares the rows in the local SQLite `Loan_table` and the target
`Loan_table` referenced by the `DATABASE_URL` env var. It is *skipped* unless
`DATABASE_URL` is set so it can be run locally or in CI where a target DB is
available.

Run locally (PowerShell):
  $env:DATABASE_URL = "mysql+pymysql://root:pass@127.0.0.1:3306/lon-default"
  .venv/Scripts/pytest -q tests/test_migration_verification.py::test_migration_verification

"""
from __future__ import annotations

import os
import math
import pytest
from decimal import Decimal
from sqlalchemy import create_engine, MetaData, Table, select


def fetch_rows(engine, table_name: str):
    metadata = MetaData()
    t = Table(table_name, metadata, autoload_with=engine)
    with engine.connect() as conn:
        rows = conn.execute(select(t)).mappings().all()
    return [dict(r) for r in rows]


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="DATABASE_URL not set")
def test_migration_verification():
    src_engine = create_engine("sqlite:///data/predictions.db")
    tgt_engine = create_engine(os.environ["DATABASE_URL"])

    src_rows = fetch_rows(src_engine, "Loan_table")
    tgt_rows = fetch_rows(tgt_engine, "Loan_table")

    assert len(src_rows) > 0, "source table must contain rows for test"

    # Map source rows by source id
    src_by_id = {r["id"]: r for r in src_rows}

    # Detect target PK name
    tgt_table = Table("Loan_table", MetaData(), autoload_with=tgt_engine)
    pk_cols = [c.name for c in tgt_table.primary_key.columns] if getattr(tgt_table, "primary_key", None) else []
    tgt_pk = pk_cols[0] if pk_cols else "id"

    # Build target map using PK or fallback to 'id'
    tgt_by_id = {}
    for r in tgt_rows:
        if tgt_pk in r:
            tgt_by_id[r[tgt_pk]] = r
        elif "id" in r:
            tgt_by_id[r["id"]] = r

    # All source IDs should be present in target (we expect migration/preserve-ids behavior)
    missing_ids = sorted(set(src_by_id.keys()) - set(tgt_by_id.keys()))
    assert not missing_ids, f"Missing source IDs in target: {missing_ids}"

    # Compare overlapping columns (case-insensitive keys)
    mismatch_rows = []
    common_ids = sorted(set(src_by_id.keys()) & set(tgt_by_id.keys()))
    for idv in common_ids:
        s = src_by_id[idv]
        t = tgt_by_id[idv]
        s_lower = {k.lower(): v for k, v in s.items()}
        t_lower = {k.lower(): v for k, v in t.items()}
        keys = set(s_lower.keys()) & set(t_lower.keys())
        diffs = []
        for k in sorted(keys):
            sv = s_lower.get(k)
            tv = t_lower.get(k)
            if sv is None and tv is None:
                continue
            try:
                # Treat Decimal as numeric for tolerant comparisons
                numeric_types = (int, float, Decimal)
                if isinstance(sv, numeric_types) and isinstance(tv, numeric_types):
                    if not math.isclose(float(sv), float(tv), rel_tol=1e-9, abs_tol=1e-9):
                        diffs.append((k, sv, tv))
                else:
                    if str(sv).strip() != str(tv).strip():
                        diffs.append((k, sv, tv))
            except Exception:
                if str(sv).strip() != str(tv).strip():
                    diffs.append((k, sv, tv))
        if diffs:
            mismatch_rows.append((idv, diffs))

    assert not mismatch_rows, f"Found {len(mismatch_rows)} rows with differences (sample: {mismatch_rows[:3]})"
