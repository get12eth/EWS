"""migrate_sqlite_to_mysql.py

Safely copy rows from the local SQLite `Loan_table` (default dev DB) to a MySQL
`Loan_table` specified by DATABASE_URL environment variable.

Usage:
  # Dry run (shows rows to insert and collision summary):
  python scripts/migrate_sqlite_to_mysql.py --dry-run

  # Commit the changes (will INSERT rows that don't exist in target):
  python scripts/migrate_sqlite_to_mysql.py --commit

  # Provide explicit source/target DB URLs:
  python scripts/migrate_sqlite_to_mysql.py --source sqlite:///data/predictions.db --target "mysql+pymysql://root:Bant%406963@127.0.0.1:3306/lon-default" --commit

Notes:
- This script will not overwrite existing rows in the target by default. It will skip
  rows where the primary key `id` already exists. Use `--preserve-ids` if you want
  to keep source IDs and ensure no collisions.
- Always run with --dry-run first. Ensure you have backups.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Dict
from sqlalchemy import create_engine, MetaData, Table, select, insert
from sqlalchemy.engine import Engine


def make_engine(url: str) -> Engine:
    return create_engine(url)


def fetch_rows(engine: Engine, table_name: str) -> List[Dict]:
    metadata = MetaData()
    t = Table(table_name, metadata, autoload_with=engine)
    with engine.connect() as conn:
        rows = conn.execute(select(t)).mappings().all()
    return [dict(r) for r in rows]


def insert_rows(engine: Engine, table_name: str, rows: List[Dict], preserve_ids: bool = False) -> Dict:
    metadata = MetaData()
    t = Table(table_name, metadata, autoload_with=engine)

    # Determine primary key column name(s) on the target table
    pk_cols = [c.name for c in t.primary_key.columns] if getattr(t, 'primary_key', None) else []
    pk = pk_cols[0] if pk_cols else 'id'

    inserted = 0
    skipped = 0
    with engine.begin() as conn:
        for r in rows:
            row_to_insert = dict(r)

            # Preserve source ID if requested. Map source 'id' to the target pk name when needed.
            source_id = r.get('id') if r is not None else None
            if preserve_ids and source_id is not None:
                # Check for existing primary key in target
                existing = conn.execute(select(getattr(t.c, pk)).where(getattr(t.c, pk) == source_id)).first()
                if existing:
                    skipped += 1
                    continue
                # Ensure the row uses the target pk column name
                if pk != 'id':
                    row_to_insert[pk] = source_id
                    row_to_insert.pop('id', None)
            else:
                # Do not preserve IDs: remove the target pk (and common 'id' key) so auto-increment can assign
                row_to_insert.pop(pk, None)
                row_to_insert.pop('id', None)

            # Filter out any columns that do not exist in the target table to avoid insertion errors
            allowed_cols = {c.name for c in t.columns}
            filtered_row = {k: v for k, v in row_to_insert.items() if k in allowed_cols}

            conn.execute(insert(t).values(**filtered_row))
            inserted += 1
    return {'inserted': inserted, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=os.environ.get('SOURCE_DB_URL', 'sqlite:///data/predictions.db'))
    parser.add_argument('--target', default=os.environ.get('DATABASE_URL'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--commit', action='store_true')
    parser.add_argument('--preserve-ids', action='store_true', help='Preserve source ids when inserting into target')
    args = parser.parse_args()

    if not args.target:
        print('ERROR: target DB URL not provided. Set --target or DATABASE_URL env var.')
        sys.exit(2)

    print('Source:', args.source)
    print('Target:', args.target)

    src_engine = make_engine(args.source)
    tgt_engine = make_engine(args.target)

    # Basic connectivity checks
    try:
        # Ensure Loan_table exists in both dbs
        src_rows = fetch_rows(src_engine, 'Loan_table')
        print(f'Found {len(src_rows)} rows in source')
    except Exception as e:
        print('ERROR reading source Loan_table:', e)
        sys.exit(2)

    try:
        tgt_rows = fetch_rows(tgt_engine, 'Loan_table')
        print(f'Found {len(tgt_rows)} rows in target')
    except Exception as e:
        print('ERROR reading target Loan_table:', e)
        sys.exit(2)

    # Prepare rows to insert (skip rows already present by id).
    # Robustly detect the target primary key name (e.g., 'id' vs 'ID') and use it when checking collisions.
    src_by_id = {r['id']: r for r in src_rows}

    metadata = MetaData()
    tgt_table = Table('Loan_table', metadata, autoload_with=tgt_engine)
    tgt_pk_cols = [c.name for c in tgt_table.primary_key.columns] if getattr(tgt_table, 'primary_key', None) else []
    tgt_pk = tgt_pk_cols[0] if tgt_pk_cols else 'id'

    tgt_ids = set()
    for r in tgt_rows:
        if tgt_pk in r:
            tgt_ids.add(r[tgt_pk])
        elif 'id' in r:
            tgt_ids.add(r['id'])

    rows_to_insert = [r for id, r in src_by_id.items() if (args.preserve_ids and id not in tgt_ids) or (not args.preserve_ids)]

    print(f'Rows considered for insertion: {len(rows_to_insert)} (preserve_ids={args.preserve_ids})')

    if args.dry_run or not args.commit:
        print('\nDRY RUN mode - no changes will be made')
        # show summary and sample
        sample = rows_to_insert[:5]
        for s in sample:
            print('- sample id', s.get('id'))
        print('\nDry-run complete. To apply changes re-run with --commit')
        return

    # Commit mode
    print('\nCommitting inserts to target...')
    result = insert_rows(tgt_engine, 'Loan_table', rows_to_insert, preserve_ids=args.preserve_ids)
    print('Insert result:', result)
    print('Done.')


if __name__ == '__main__':
    main()
