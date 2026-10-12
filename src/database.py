import sqlite3
from pathlib import Path
from datetime import datetime

# Resolve the database path relative to the project root (parent of src/).
# If database.py is in src/database.py, this points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "documind.db"


def get_connection():
    """Return a SQLite connection and ensure the database directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create required database tables if they do not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                document_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                pages INTEGER NOT NULL DEFAULT 0,
                chunks INTEGER NOT NULL DEFAULT 0,
                processing_time REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_document(
    filename,
    document_type,
    confidence,
    pages,
    chunks,
    processing_time,
):
    """Save one processed document record."""
    initialize_database()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                filename,
                document_type,
                confidence,
                pages,
                chunks,
                processing_time,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                document_type,
                float(confidence or 0),
                int(pages or 0),
                int(chunks or 0),
                processing_time,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_documents():
    """Return saved document records, newest first."""
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                document_type,
                confidence,
                pages,
                chunks,
                processing_time,
                created_at
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()

    return [tuple(row) for row in rows]


def get_analytics():
    """Return document totals, averages, and document-type distribution."""
    initialize_database()

    with get_connection() as connection:
        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS total_documents,
                AVG(confidence) AS avg_confidence,
                AVG(processing_time) AS avg_processing_time
            FROM documents
            """
        ).fetchone()

        distribution = connection.execute(
            """
            SELECT document_type, COUNT(*) AS count
            FROM documents
            GROUP BY document_type
            ORDER BY count DESC
            """
        ).fetchall()

    return {
        "total_documents": summary["total_documents"] or 0,
        "avg_confidence": round(summary["avg_confidence"] or 0, 2),
        "avg_processing_time": round(
            summary["avg_processing_time"] or 0, 2
        ),
        "distribution": [
            (row["document_type"], row["count"]) for row in distribution
        ],
    }


# Ensure the schema exists as soon as this module is imported.
initialize_database()
