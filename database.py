
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "phishing_detection.db"


class Database:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS emails (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject     TEXT,
                    body        TEXT NOT NULL,
                    source      TEXT DEFAULT 'cli',
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id        INTEGER NOT NULL,
                    prediction      TEXT NOT NULL,      -- 'phishing' / 'legitimate'
                    confidence      REAL NOT NULL,       -- 0..1
                    features_json   TEXT NOT NULL,       -- extracted feature snapshot
                    model_version   TEXT,
                    analyzed_at     TEXT NOT NULL,
                    FOREIGN KEY (email_id) REFERENCES emails (id)
                );

                CREATE TABLE IF NOT EXISTS phishing_attempts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id        INTEGER NOT NULL,
                    result_id       INTEGER NOT NULL,
                    subject         TEXT,
                    confidence      REAL NOT NULL,
                    detected_at     TEXT NOT NULL,
                    FOREIGN KEY (email_id) REFERENCES emails (id),
                    FOREIGN KEY (result_id) REFERENCES analysis_results (id)
                );

                CREATE TABLE IF NOT EXISTS model_metadata (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version   TEXT NOT NULL,
                    algorithm       TEXT NOT NULL,
                    accuracy        REAL,
                    precision_score REAL,
                    recall_score    REAL,
                    f1_score        REAL,
                    trained_at      TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------
    def add_email(self, body, subject=None, source="cli"):
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO emails (subject, body, source, created_at) "
                "VALUES (?, ?, ?, ?)",
                (subject, body, source, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def add_analysis_result(self, email_id, prediction, confidence,
                             features, model_version=None):
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO analysis_results "
                "(email_id, prediction, confidence, features_json, "
                " model_version, analyzed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    email_id,
                    prediction,
                    confidence,
                    json.dumps(features),
                    model_version,
                    datetime.utcnow().isoformat(),
                ),
            )
            result_id = cur.lastrowid

            if prediction == "phishing":
                subject_row = conn.execute(
                    "SELECT subject FROM emails WHERE id = ?", (email_id,)
                ).fetchone()
                subject = subject_row["subject"] if subject_row else None
                conn.execute(
                    "INSERT INTO phishing_attempts "
                    "(email_id, result_id, subject, confidence, detected_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        email_id,
                        result_id,
                        subject,
                        confidence,
                        datetime.utcnow().isoformat(),
                    ),
                )
            return result_id

    def add_model_metadata(self, model_version, algorithm, accuracy,
                            precision_score, recall_score, f1_score):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_metadata "
                "(model_version, algorithm, accuracy, precision_score, "
                " recall_score, f1_score, trained_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_version,
                    algorithm,
                    accuracy,
                    precision_score,
                    recall_score,
                    f1_score,
                    datetime.utcnow().isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_recent_results(self, limit=20):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.id, e.subject, r.prediction, r.confidence, r.analyzed_at
                FROM analysis_results r
                JOIN emails e ON e.id = r.email_id
                ORDER BY r.analyzed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_phishing_attempts(self, limit=50):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, email_id, subject, confidence, detected_at
                FROM phishing_attempts
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM analysis_results").fetchone()["c"]
            phishing = conn.execute(
                "SELECT COUNT(*) AS c FROM analysis_results WHERE prediction = 'phishing'"
            ).fetchone()["c"]
            legitimate = total - phishing
            avg_conf = conn.execute(
                "SELECT AVG(confidence) AS a FROM analysis_results"
            ).fetchone()["a"]
            latest_model = conn.execute(
                "SELECT * FROM model_metadata ORDER BY trained_at DESC LIMIT 1"
            ).fetchone()
            return {
                "total_analyzed": total,
                "phishing_detected": phishing,
                "legitimate": legitimate,
                "average_confidence": round(avg_conf, 4) if avg_conf else None,
                "latest_model": dict(latest_model) if latest_model else None,
            }

    def get_email_features(self, email_id):
        """Return stored features for an email (used for future retraining)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT features_json FROM analysis_results WHERE email_id = ? "
                "ORDER BY analyzed_at DESC LIMIT 1",
                (email_id,),
            ).fetchone()
            return json.loads(row["features_json"]) if row else None
