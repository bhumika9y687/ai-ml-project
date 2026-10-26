import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from utils import database


class DatabaseRoleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "cardiovue.db")
        self.db_path_patch = patch.object(database, "DB_PATH", self.db_path)
        self.db_path_patch.start()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_registration_and_authentication_support_all_roles(self):
        database.init_db()

        for role in ("patient", "doctor", "researcher"):
            username = f"{role}_account"
            created, _ = database.register_user(
                username, "password123", role.title(), f"{role}@example.com", role=role
            )

            self.assertTrue(created)
            self.assertEqual(database.authenticate(username, "password123")["role"], role)

    def test_migrates_patient_only_constraint_and_preserves_existing_data(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role = 'patient'),
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        extra_json TEXT DEFAULT '{}',
                        joined TEXT NOT NULL,
                        family_history INTEGER DEFAULT 0
                    );
                    CREATE TABLE health_records (
                        id INTEGER PRIMARY KEY,
                        username TEXT NOT NULL REFERENCES users(username),
                        marker TEXT
                    );
                    """
                )
                conn.execute(
                    """INSERT INTO users
                       (username, password_hash, role, name, email, joined)
                       VALUES (?, ?, 'patient', ?, ?, ?)""",
                    ("existing", database._hash("password123"), "Existing", "e@example.com", "2026-01-01"),
                )
                conn.execute(
                    "INSERT INTO health_records (username, marker) VALUES (?, ?)",
                    ("existing", "preserve me"),
                )

        database.init_db()

        self.assertEqual(database.authenticate("existing", "password123")["role"], "patient")
        created, _ = database.register_user(
            "doctor_account", "password123", "Doctor", "doctor@example.com", role="doctor"
        )
        self.assertTrue(created)
        with closing(sqlite3.connect(self.db_path)) as conn:
            marker = conn.execute(
                "SELECT marker FROM health_records WHERE username = 'existing'"
            ).fetchone()[0]
        self.assertEqual(marker, "preserve me")


if __name__ == "__main__":
    unittest.main()
