"""Backfill first-version provenance without resetting the database.

By default this is a dry run.  ``--apply`` writes only first-version metadata;
existing machine mentions remain untouched and stay excluded from alpha because
they have no version-1 methods hash.  Add ``--queue-reprocess`` to replace the
old mentions with extraction from the version-1 full-text page.

Run from the repository root:
    DATABASE_URL=... python scripts/backfill_first_version_metadata.py --apply
    DATABASE_URL=... python scripts/backfill_first_version_metadata.py --apply --queue-reprocess
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_db_connection
from src.main import fetch_first_version_metadata, init_db


logger = logging.getLogger(__name__)


def candidates_needing_backfill(limit: int | None, queue_reprocess: bool) -> list[tuple[str, str]]:
    query = """
        SELECT doi, COALESCE(NULLIF(server, ''), 'biorxiv')
        FROM candidates
        WHERE first_version IS NULL
           OR first_version_date IS NULL
           OR source_version_url IS NULL
           OR first_version_api_url IS NULL
        ORDER BY date, doi
    """
    if queue_reprocess:
        # After a metadata-only run, legacy rows are fully described but still
        # lack a hash proving their mentions came from version 1.
        query = query.replace("ORDER BY", "OR methods_sha256 IS NULL\n        ORDER BY")
    if limit is not None:
        query += " LIMIT %s"
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if limit is None:
                cursor.execute(query)
            else:
                cursor.execute(query, (limit,))
            return cursor.fetchall()
    finally:
        conn.close()


def write_metadata(metadata: dict, queue_reprocess: bool) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE candidates
                SET title = %s,
                    category = %s,
                    date = %s,
                    abstract = %s,
                    server = %s,
                    first_version = %s,
                    first_version_date = %s,
                    source_version_url = %s,
                    first_version_api_url = %s
                WHERE doi = %s
                """,
                (
                    metadata["title"], metadata["category"], metadata["date"],
                    metadata["abstract"], metadata["server"], metadata["version"],
                    metadata["date"], metadata["source_version_url"],
                    metadata["first_version_api_url"], metadata["doi"],
                ),
            )
            if queue_reprocess:
                # The old records could have been extracted from a later
                # revision.  Delete only this paper's derived mentions before
                # re-queuing it; source candidates and run history are kept.
                cursor.execute("DELETE FROM machine_mentions WHERE doi = %s", (metadata["doi"],))
                cursor.execute(
                    """
                    UPDATE candidates
                    SET processed = 0,
                        processed_at = NULL,
                        methods_sha256 = NULL,
                        methods_fetched_at = NULL
                    WHERE doi = %s
                    """,
                    (metadata["doi"],),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write metadata; otherwise only report planned changes.")
    parser.add_argument(
        "--queue-reprocess",
        action="store_true",
        help="Delete each migrated paper's old mentions and queue version-1 re-extraction (requires --apply).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many candidates.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between API requests (default: 0.4).")
    args = parser.parse_args()
    if args.queue_reprocess and not args.apply:
        parser.error("--queue-reprocess requires --apply")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()
    candidates = candidates_needing_backfill(args.limit, args.queue_reprocess)
    mode = "apply" if args.apply else "dry run"
    logger.info("%s: %d candidate(s) need first-version metadata", mode, len(candidates))

    session = requests.Session()
    migrated = failures = 0
    for index, (doi, server) in enumerate(candidates, start=1):
        try:
            metadata = fetch_first_version_metadata(session, doi, server)
            if args.apply:
                write_metadata(metadata, args.queue_reprocess)
            migrated += 1
            logger.info("%s v%s → %s", doi, metadata["version"], metadata["date"])
        except Exception as exc:
            failures += 1
            logger.warning("%s: %s", doi, exc)
        if index < len(candidates):
            time.sleep(args.delay)

    logger.info("Completed: %d migrated, %d failed", migrated, failures)
    if args.apply and not args.queue_reprocess:
        logger.info("Mentions were retained and remain excluded from alpha until version-1 reprocessing.")
    if args.queue_reprocess:
        logger.info("Run `python -m src.main --append` to process the queued candidates.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
