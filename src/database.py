import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = Path("data/documind.db")


def get_connection():
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return sqlite3.connect(
        DB_PATH
    )


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            pages INTEGER NOT NULL,
            chunks INTEGER NOT NULL,
            processing_time REAL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_document(
    filename,
    document_type,
    confidence,
    pages,
    chunks,
    processing_time
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
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
            confidence,
            pages,
            chunks,
            processing_time,
            datetime.now().isoformat(
                timespec="seconds"
            ),
        )
    )

    connection.commit()
    connection.close()


def get_documents():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
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
    )

    rows = cursor.fetchall()

    connection.close()

    return rows
def get_analytics():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_documents,
            AVG(confidence) AS avg_confidence,
            AVG(processing_time) AS avg_processing_time
        FROM documents
        """
    )

    summary = cursor.fetchone()

    cursor.execute(
        """
        SELECT
            document_type,
            COUNT(*) AS count
        FROM documents
        GROUP BY document_type
        ORDER BY count DESC
        """
    )

    distribution = cursor.fetchall()

    connection.close()

    return {
        "total_documents": summary[0] or 0,
        "avg_confidence": round(
            summary[1] or 0,
            2
        ),
        "avg_processing_time": round(
            summary[2] or 0,
            2
        ),
        "distribution": distribution,
    }