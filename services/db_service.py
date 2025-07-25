# services/db_service.py

import psycopg2


def get_db():
    return psycopg2.connect(
        dbname="voice_db",
        user="postgres",
        password="moni123",
        host="localhost"
    )

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    id SERIAL PRIMARY KEY,
                    language TEXT NOT NULL,
                    voice_name TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS finetuning_profiles (
                    id SERIAL PRIMARY KEY,
                    language TEXT NOT NULL,
                    voice_name TEXT NOT NULL,
                    audio_path TEXT NOT NULL
                )
            """)
            conn.commit()