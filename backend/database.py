import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    """Connect to PostgreSQL using the Render DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    return psycopg2.connect(database_url)


def init_database():
    """Create the analysis_history table if it does not exist."""
    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    prediction TEXT,
                    is_human BOOLEAN,
                    confidence DOUBLE PRECISION,
                    demo_mode BOOLEAN,
                    model_used TEXT,
                    status_note TEXT,
                    duration DOUBLE PRECISION,
                    sampling_rate INTEGER,
                    processing_time_ms DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        conn.commit()
        print("PostgreSQL database initialized successfully.")

    except Exception as e:
        print(f"Database initialization error: {e}")

    finally:
        if conn:
            conn.close()


def save_analysis(
    filename,
    prediction,
    is_human,
    confidence,
    demo_mode,
    model_used,
    status_note,
    duration,
    sampling_rate,
    processing_time_ms
):
    """Save one completed voice analysis."""

    conn = None

    try:
        conn = get_connection()

        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO analysis_history (
                    filename,
                    prediction,
                    is_human,
                    confidence,
                    demo_mode,
                    model_used,
                    status_note,
                    duration,
                    sampling_rate,
                    processing_time_ms
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                RETURNING id;
            """, (
                filename,
                prediction,
                is_human,
                confidence,
                demo_mode,
                model_used,
                status_note,
                duration,
                sampling_rate,
                processing_time_ms
            ))

            analysis_id = cursor.fetchone()[0]

        conn.commit()

        return analysis_id

    except Exception as e:
        if conn:
            conn.rollback()

        print(f"Database save error: {e}")
        return None

    finally:
        if conn:
            conn.close()


def get_analysis_history(limit=50):
    """Return recent analysis records."""

    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    id,
                    filename,
                    prediction,
                    is_human,
                    confidence,
                    demo_mode,
                    model_used,
                    status_note,
                    duration,
                    sampling_rate,
                    processing_time_ms,
                    created_at
                FROM analysis_history
                ORDER BY created_at DESC
                LIMIT %s;
            """, (limit,))

            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    except Exception as e:
        print(f"Database history error: {e}")
        return []

    finally:
        if conn:
            conn.close()