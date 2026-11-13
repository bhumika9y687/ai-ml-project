"""SQLite persistence for patient accounts and risk-assessment history."""

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("CARDIOVUE_DB", "cardiovue.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the tables needed by the patient assessment workflow."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor', 'researcher')),
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                extra_json TEXT DEFAULT '{}',
                joined TEXT NOT NULL,
                family_history INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                date TEXT NOT NULL,
                age INTEGER,
                bmi REAL,
                highbp INTEGER DEFAULT 0,
                highchol INTEGER DEFAULT 0,
                smoker INTEGER DEFAULT 0,
                diabetes INTEGER DEFAULT 0,
                phys_activity INTEGER DEFAULT 0,
                gen_health INTEGER DEFAULT 3,
                risk_score REAL,
                risk_label TEXT,
                cholesterol INTEGER,
                bp_systolic INTEGER,
                bp_diastolic INTEGER,
                heart_rate INTEGER,
                notes TEXT DEFAULT '',
                shap_json TEXT DEFAULT '{}',
                model_used TEXT DEFAULT 'Ensemble',
                family_history INTEGER DEFAULT 0,
                FOREIGN KEY(username) REFERENCES users(username)
            );
            """
        )
        _migrate_user_roles(conn)
    _seed_demo_patient()


def _migrate_user_roles(conn):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    if row is None or not re.search(
        r"CHECK\s*\(\s*role\s*=\s*'patient'\s*\)", row["sql"], re.IGNORECASE
    ):
        return

    conn.executescript(
        """
        CREATE TABLE users_new (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('patient', 'doctor', 'researcher')),
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            extra_json TEXT DEFAULT '{}',
            joined TEXT NOT NULL,
            family_history INTEGER DEFAULT 0
        );
        INSERT INTO users_new
            (username, password_hash, role, name, email, extra_json, joined, family_history)
        SELECT username, password_hash, role, name, email, extra_json, joined, family_history
        FROM users;
        DROP TABLE users;
        ALTER TABLE users_new RENAME TO users;
        """
    )


def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _seed_demo_patient():
    with get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        if exists:
            return
        conn.execute(
            """INSERT INTO users
               (username, password_hash, role, name, email, extra_json, joined)
               VALUES (?, ?, 'patient', ?, ?, ?, ?)""",
            (
                "patient1",
                _hash("patient123"),
                "Demo Patient",
                "patient@example.com",
                json.dumps({"age": 45}),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        )


def get_user(username):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    user = dict(row)
    try:
        user.update(json.loads(user.get("extra_json") or "{}"))
    except (json.JSONDecodeError, TypeError):
        pass
    return user


def authenticate(username, password):
    user = get_user(username)
    if (
        user
        and user.get("role") in {"patient", "doctor", "researcher"}
        and user.get("password_hash") == _hash(password)
    ):
        return user
    return None


def register_user(username, password, name, email, extra=None, role="patient"):
    """Create a patient, doctor, or researcher account."""
    if role not in {"patient", "doctor", "researcher"}:
        raise ValueError("Role must be patient, doctor, or researcher.")
    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO users
                   (username, password_hash, role, name, email, extra_json, joined)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    username,
                    _hash(password),
                    role,
                    name,
                    email,
                    json.dumps(extra or {}),
                    datetime.now().strftime("%Y-%m-%d"),
                ),
            )
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError as exc:
        if "users.username" in str(exc):
            return False, "Username already exists."
        raise


def get_health_records(username, limit=100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM health_records WHERE username = ? ORDER BY date DESC, id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    records = [dict(row) for row in rows]
    for record in records:
        try:
            record["shap_values"] = json.loads(record.pop("shap_json", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            record["shap_values"] = {}
    return records


def insert_health_record(username, data: dict):
    columns = [
        "username", "date", "age", "bmi", "highbp", "highchol", "smoker",
        "diabetes", "phys_activity", "gen_health", "risk_score", "risk_label",
        "cholesterol", "bp_systolic", "bp_diastolic", "heart_rate",
        "shap_json", "model_used", "notes",
    ]
    values = [
        username,
        data.get("date", datetime.now().strftime("%Y-%m-%d")),
        data.get("age"),
        data.get("bmi"),
        int(data.get("highbp", 0)),
        int(data.get("highchol", 0)),
        int(data.get("smoker", 0)),
        int(data.get("diabetes", 0)),
        int(data.get("phys_activity", 0)),
        int(data.get("gen_health", 3)),
        data.get("risk_score"),
        data.get("risk_label"),
        data.get("cholesterol"),
        data.get("bp_systolic"),
        data.get("bp_diastolic"),
        data.get("heart_rate"),
        json.dumps(data.get("shap_values", {})),
        data.get("model_used", "Ensemble"),
        data.get("notes", ""),
    ]
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO health_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            values,
        )
