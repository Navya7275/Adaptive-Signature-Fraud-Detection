"""
SQLite database manager.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from app.config import DB_PATH

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db():
    """Create tables from schema.sql and apply lightweight migrations."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())

    # Migration: add columns that CREATE TABLE IF NOT EXISTS won't add
    # to databases created before the column existed.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(drift_profiles)")}
    if "base_threshold" not in cols:
        conn.execute(
            "ALTER TABLE drift_profiles ADD COLUMN base_threshold REAL DEFAULT 0.80")
        conn.commit()

    sig_cols = {r[1] for r in conn.execute("PRAGMA table_info(signatures)")}
    if "file_hash" not in sig_cols:
        conn.execute("ALTER TABLE signatures ADD COLUMN file_hash TEXT")
        conn.commit()

    # Backfill hashes for rows created before the column existed, so
    # duplicate detection also covers previously stored images.
    import hashlib
    for sig_id, path in conn.execute(
            "SELECT id, image_path FROM signatures WHERE file_hash IS NULL").fetchall():
        try:
            digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
            conn.execute("UPDATE signatures SET file_hash = ? WHERE id = ?",
                         (digest, sig_id))
        except OSError:
            pass  # image file missing — leave NULL
    conn.commit()

    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


@contextmanager
def get_db():
    """Yields a connection with Row factory. Auto-commits or rolls back."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row):
    return dict(row) if row else None


def rows_to_dicts(rows):
    return [dict(r) for r in rows]