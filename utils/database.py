"""
Cardio Vue AI — SQLite Persistence Layer
Replaces session_state mock DB with real persistent storage.
"""

import sqlite3
import hashlib
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager
import numpy as np

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
    """Create all tables on first run."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username     TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role         TEXT NOT NULL CHECK(role IN ('patient','doctor','researcher')),
            name         TEXT NOT NULL,
            email        TEXT NOT NULL,
            extra_json   TEXT DEFAULT '{}',
            joined       TEXT NOT NULL,
            family_history INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS health_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT NOT NULL,
            date         TEXT NOT NULL,
            age          INTEGER,
            bmi          REAL,
            highbp       INTEGER DEFAULT 0,
            highchol     INTEGER DEFAULT 0,
            smoker       INTEGER DEFAULT 0,
            diabetes     INTEGER DEFAULT 0,
            phys_activity INTEGER DEFAULT 0,
            gen_health   INTEGER DEFAULT 3,
            risk_score   REAL,
            risk_label   TEXT,
            cholesterol  INTEGER,
            bp_systolic  INTEGER,
            bp_diastolic INTEGER,
            heart_rate   INTEGER,
            notes        TEXT DEFAULT '',
            shap_json    TEXT DEFAULT '{}',
            model_used   TEXT DEFAULT 'Ensemble',
            family_history INTEGER DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username)
        );

        CREATE TABLE IF NOT EXISTS appointments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            doctor_user  TEXT NOT NULL,
            doctor_name  TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            type         TEXT NOT NULL,
            status       TEXT DEFAULT 'pending',
            notes        TEXT DEFAULT '',
            FOREIGN KEY(patient_user) REFERENCES users(username)
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT NOT NULL,
            type         TEXT NOT NULL,
            msg          TEXT NOT NULL,
            time_str     TEXT NOT NULL,
            is_read      INTEGER DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username)
        );

        CREATE TABLE IF NOT EXISTS blood_tests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT NOT NULL,
            date         TEXT NOT NULL,
            hdl          REAL, ldl REAL, triglycerides REAL,
            glucose      REAL, hba1c REAL, creatinine REAL,
            notes        TEXT DEFAULT '',
            FOREIGN KEY(username) REFERENCES users(username)
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            room         TEXT NOT NULL,
            sender       TEXT NOT NULL,
            sender_name  TEXT NOT NULL,
            message      TEXT NOT NULL,
            timestamp    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS goals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT NOT NULL,
            goal_type    TEXT NOT NULL,
            target_value REAL NOT NULL,
            current_value REAL,
            start_date   TEXT NOT NULL,
            target_date  TEXT NOT NULL,
            status       TEXT DEFAULT 'active',
            achieved_date TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        );

        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_username TEXT NOT NULL,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            status TEXT NOT NULL,
            f1 REAL,
            auc REAL,
            parameters TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(researcher_username) REFERENCES users(username)
        );
        """)


def _hash(pw): 
    return hashlib.sha256(pw.encode()).hexdigest()


def _risk_label(score):
    if score < 25: return "Low"
    elif score < 50: return "Moderate"
    elif score < 75: return "High"
    else: return "Critical"


def _seed_demo_data():
    """Insert demo users and records if DB is empty."""
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if existing > 0:
            return  # Already seeded

        users = [
            ("patient1",  _hash("patient123"),  "patient",    "Batman",      "batman@email.com",       '{"age":45,"gender":"Male"}',         "2024-01-15"),
            ("patient2",  _hash("patient123"),  "patient",    "Meera Iyer",        "meera@email.com",      '{"age":62,"gender":"Female"}',        "2024-03-02"),
            ("patient3",  _hash("patient123"),  "patient",    "Rohan Das",         "rohan@email.com",      '{"age":38,"gender":"Male"}',          "2024-05-10"),
            ("doctor1",   _hash("doctor123"),   "doctor",     "Dr. Kishan",        "kishan@hospital.com",   '{"specialty":"Cardiologist"}',        "2023-06-01"),
            ("doctor2",   _hash("doctor123"),   "doctor",     "Dr. Amit Kumar",    "amit@hospital.com",    '{"specialty":"Internal Medicine"}',   "2023-08-15"),
            ("researcher1",_hash("research123"),"researcher", "Ekta",   "ekta@institute.edu",  '{"institution":"AIIMS Research"}',    "2023-09-10"),
        ]
        conn.executemany(
            "INSERT INTO users (username, password_hash, role, name, email, extra_json, joined) VALUES (?,?,?,?,?,?,?)",
            users
        )

        # Seed health records for patient1 — 24-week trend
        np.random.seed(42)
        dates = [
            (datetime.now() - timedelta(weeks=23 - i)).strftime("%Y-%m-%d")
            for i in range(24)
        ]
        risk_trend = np.linspace(74, 52, 24) + np.random.normal(0, 2.5, 24)
        for i, d in enumerate(dates):
            r = float(risk_trend[i])
            lbl = _risk_label(r)
            conn.execute(
                """INSERT INTO health_records
                   (username,date,age,bmi,highbp,highchol,smoker,diabetes,
                    phys_activity,gen_health,risk_score,risk_label,
                    cholesterol,bp_systolic,bp_diastolic,heart_rate,
                    shap_json,model_used)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("patient1", d, 45,
                 round(27.5 + np.random.normal(0, 0.4), 1),
                 1, 1, 0, 0, 1, 3,
                 round(r, 1), lbl,
                 int(200 + np.random.normal(0, 12)),
                 int(128 + np.random.normal(0, 5)),
                 int(82 + np.random.normal(0, 3)),
                 int(72 + np.random.normal(0, 4)),
                 '{"Diabetes":22,"HighBP":20,"Smoking":0,"BMI":8,"Age":18}',
                 "Stacking Ensemble")
            )

        # Seed patient2 records
        risk2 = np.linspace(82, 68, 12) + np.random.normal(0, 2, 12)
        for i in range(12):
            d = (datetime.now() - timedelta(weeks=11 - i)).strftime("%Y-%m-%d")
            r = float(risk2[i])
            conn.execute(
                """INSERT INTO health_records
                   (username,date,age,bmi,highbp,highchol,smoker,diabetes,
                    phys_activity,gen_health,risk_score,risk_label,
                    cholesterol,bp_systolic,bp_diastolic,heart_rate,model_used)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("patient2", d, 62,
                 round(31.2 + np.random.normal(0, 0.3), 1),
                 1, 1, 1, 1, 0, 4,
                 round(r, 1), _risk_label(r),
                 int(230 + np.random.normal(0, 15)),
                 int(145 + np.random.normal(0, 6)),
                 int(92 + np.random.normal(0, 4)),
                 int(80 + np.random.normal(0, 5)),
                 "XGBoost")
            )

        # Appointments - UPDATED with Batman and Dr. Kishan
        apts = [
            ("patient1","Batman","doctor1","Dr. Kishan",
             (datetime.now()+timedelta(days=5)).strftime("%Y-%m-%d"),
             "10:30 AM","Cardiology Review","confirmed","Follow-up on BP medication"),
            ("patient1","Batman","doctor1","Dr. Kishan",
             (datetime.now()-timedelta(days=14)).strftime("%Y-%m-%d"),
             "2:00 PM","ECG Review","completed","ECG within normal range"),
            ("patient2","Meera Iyer","doctor1","Dr. Kishan",
             (datetime.now()+timedelta(days=2)).strftime("%Y-%m-%d"),
             "11:00 AM","Risk Assessment","confirmed","Post-discharge follow-up"),
            ("patient3","Rohan Das","doctor2","Dr. Amit Kumar",
             (datetime.now()+timedelta(days=8)).strftime("%Y-%m-%d"),
             "3:00 PM","General Check-up","pending","First consultation"),
        ]
        conn.executemany(
            "INSERT INTO appointments(patient_user,patient_name,doctor_user,doctor_name,date,time,type,status,notes) VALUES(?,?,?,?,?,?,?,?,?)",
            apts
        )

        # Notifications - UPDATED with Batman and Dr. Kishan
        notifs = [
            ("patient1","reminder","Take Aspirin 75mg – 8:00 AM","2 hours ago",0),
            ("patient1","alert","Risk score improved 14pts this month! Keep going.","1 day ago",0),
            ("patient1","appointment","Appointment with Dr. Kishan in 5 days","2 days ago",1),
            ("doctor1","alert","Patient Batman: risk score flagged — review recommended","3 hours ago",0),
            ("doctor1","alert","Patient Meera Iyer: Critical risk (78%) — urgent review","1 hour ago",0),
            ("doctor1","appointment","Upcoming: Batman – 10:30 AM in 5 days","1 day ago",1),
        ]
        conn.executemany(
            "INSERT INTO notifications(username,type,msg,time_str,is_read) VALUES(?,?,?,?,?)",
            notifs
        )

        # Blood tests for patient1
        conn.execute(
            """INSERT INTO blood_tests(username,date,hdl,ldl,triglycerides,glucose,hba1c,creatinine)
               VALUES(?,?,?,?,?,?,?,?)""",
            ("patient1", datetime.now().strftime("%Y-%m-%d"),
             52, 118, 145, 96, 5.4, 0.92)
        )


# ─── USER CRUD ─────────────────────────────────────────────────────────────────

def get_user(username):
    """Get user by username, returns dict with all fields."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row:
            d = dict(row)
            # Parse extra_json safely
            extra_json = d.get("extra_json", "{}")
            if extra_json and extra_json != '{}':
                try:
                    extra = json.loads(extra_json)
                    for k, v in extra.items():
                        d[k] = v
                except:
                    pass
            return d
    return None


def authenticate(username, password):
    """Authenticate user with username and password."""
    user = get_user(username)
    if user:
        stored_hash = user.get("password_hash")
        if stored_hash and stored_hash == _hash(password):
            return user
    return None


def register_user(username, password, role, name, email, extra=None):
    """Register a new user."""
    if get_user(username):
        return False, "Username already exists."
    extra_json = json.dumps(extra or {})
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, role, name, email, extra_json, joined) VALUES(?,?,?,?,?,?,?)",
                (username, _hash(password), role, name, email,
                 extra_json, datetime.now().strftime("%Y-%m-%d"))
            )
        return True, "Account created! Please sign in."
    except Exception as e:
        return False, str(e)


# ─── HEALTH RECORDS ────────────────────────────────────────────────────────────

def get_health_records(username, limit=100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM health_records WHERE username=? ORDER BY date DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_health_record(username, data: dict):
    cols = ["username","date","age","bmi","highbp","highchol","smoker","diabetes",
            "phys_activity","gen_health","risk_score","risk_label",
            "cholesterol","bp_systolic","bp_diastolic","heart_rate",
            "shap_json","model_used","notes"]
    vals = [
        username,
        data.get("date", datetime.now().strftime("%Y-%m-%d")),
        data.get("age"), data.get("bmi"),
        int(data.get("highbp", 0)), int(data.get("highchol", 0)),
        int(data.get("smoker", 0)), int(data.get("diabetes", 0)),
        int(data.get("phys_activity", 0)), int(data.get("gen_health", 3)),
        data.get("risk_score"), data.get("risk_label"),
        data.get("cholesterol"), data.get("bp_systolic"),
        data.get("bp_diastolic"), data.get("heart_rate"),
        json.dumps(data.get("shap_values", {})),
        data.get("model_used", "Ensemble"),
        data.get("notes", ""),
    ]
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO health_records({','.join(cols)}) VALUES({','.join(['?']*len(cols))})",
            vals
        )


def get_all_patients():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role='patient'"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        extra_json = d.get("extra_json", "{}")
        if extra_json and extra_json != '{}':
            try:
                extra = json.loads(extra_json)
                for k, v in extra.items():
                    d[k] = v
            except:
                pass
        result.append(d)
    return result


def get_all_doctors():
    """Return all registered doctor accounts, including their profile details."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role='doctor' ORDER BY name COLLATE NOCASE"
        ).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        extra_json = d.get("extra_json", "{}")
        if extra_json and extra_json != '{}':
            try:
                d.update(json.loads(extra_json))
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result


def get_doctor_patients(doctor_username):
    """Return patients who have booked an appointment with this doctor."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT u.*
               FROM users u
               JOIN appointments a ON a.patient_user = u.username
               WHERE a.doctor_user = ? AND u.role = 'patient'
               ORDER BY u.name COLLATE NOCASE""",
            (doctor_username,)
        ).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        extra_json = d.get("extra_json", "{}")
        if extra_json and extra_json != '{}':
            try:
                d.update(json.loads(extra_json))
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result


def get_research_records():
    """Return de-identified health records for researcher-facing analysis."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT h.*, u.extra_json
               FROM health_records h
               JOIN users u ON u.username = h.username
               WHERE u.role = 'patient'
               ORDER BY h.date DESC, h.id DESC"""
        ).fetchall()

    records = []
    for row in rows:
        data = dict(row)
        try:
            profile = json.loads(data.pop('extra_json') or '{}')
        except (json.JSONDecodeError, TypeError):
            profile = {}
        records.append({
            'patient_id': f"P-{hashlib.sha256(data['username'].encode()).hexdigest()[:10].upper()}",
            'date': data['date'],
            'age': data.get('age') or profile.get('age'),
            'gender': profile.get('gender', 'Not recorded'),
            'bmi': data.get('bmi'),
            'highbp': data.get('highbp', 0),
            'highchol': data.get('highchol', 0),
            'smoker': data.get('smoker', 0),
            'diabetes': data.get('diabetes', 0),
            'phys_activity': data.get('phys_activity', 0),
            'risk_score': data.get('risk_score'),
            'risk_label': data.get('risk_label', 'Unknown'),
        })
    return records


def get_experiments(researcher_username):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE researcher_username=? ORDER BY id DESC",
            (researcher_username,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_experiment(researcher_username, data):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO experiments
               (researcher_username, name, model_type, status, f1, auc, parameters, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (researcher_username, data['name'], data['model_type'], data['status'],
             data.get('f1'), data.get('auc'), data.get('parameters', ''),
             data.get('notes', ''), datetime.now().strftime('%Y-%m-%d %H:%M')),
        )


# ─── APPOINTMENTS ──────────────────────────────────────────────────────────────

def get_appointments(username=None, role="patient"):
    with get_conn() as conn:
        if role == "patient":
            rows = conn.execute(
                "SELECT * FROM appointments WHERE patient_user=? ORDER BY date ASC, time ASC",
                (username,)
            ).fetchall()
        elif role == "doctor":
            rows = conn.execute(
                "SELECT * FROM appointments WHERE doctor_user=? ORDER BY date ASC",
                (username,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM appointments ORDER BY date DESC").fetchall()
    return [dict(r) for r in rows]


def book_appointment(data: dict):
    with get_conn() as conn:
        duplicate = conn.execute(
            """SELECT 1 FROM appointments
               WHERE patient_user=? AND doctor_user=? AND date=? AND time=?
                 AND status IN ('pending', 'confirmed')""",
            (data["patient_user"], data["doctor_user"], data["date"], data["time"]),
        ).fetchone()
        if duplicate:
            return False, "You already have an active appointment with this doctor at that time."
        conn.execute(
            """INSERT INTO appointments
               (patient_user,patient_name,doctor_user,doctor_name,date,time,type,status,notes)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (data["patient_user"], data["patient_name"],
             data["doctor_user"], data["doctor_name"],
             data["date"], data["time"], data["type"],
             "pending", data.get("notes",""))
        )
    return True, "Appointment request sent."


def update_appointment_status(appt_id, status):
    with get_conn() as conn:
        conn.execute("UPDATE appointments SET status=? WHERE id=?", (status, appt_id))


# ─── NOTIFICATIONS ─────────────────────────────────────────────────────────────

def get_notifications(username):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM notifications WHERE username=? ORDER BY id DESC",
            (username,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_all_read(username):
    with get_conn() as conn:
        conn.execute("UPDATE notifications SET is_read=1 WHERE username=?", (username,))


def add_notification(username, ntype, msg):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notifications(username,type,msg,time_str,is_read) VALUES(?,?,?,?,0)",
            (username, ntype, msg, "just now")
        )


# ─── BLOOD TESTS ───────────────────────────────────────────────────────────────

def get_blood_tests(username):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM blood_tests WHERE username=? ORDER BY date DESC",
            (username,)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_blood_test(username, data: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO blood_tests(username,date,hdl,ldl,triglycerides,glucose,hba1c,creatinine,notes)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (username, datetime.now().strftime("%Y-%m-%d"),
             data.get("hdl"), data.get("ldl"), data.get("triglycerides"),
             data.get("glucose"), data.get("hba1c"), data.get("creatinine"),
             data.get("notes", ""))
        )


# ─── GOALS ─────────────────────────────────────────────────────────────────────

def get_goals(username, status='active'):
    """Get goals for a user."""
    with get_conn() as conn:
        if status == 'all':
            rows = conn.execute(
                "SELECT * FROM goals WHERE username=? ORDER BY target_date",
                (username,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM goals WHERE username=? AND status=? ORDER BY target_date",
                (username, status)
            ).fetchall()
    return [dict(r) for r in rows]


def create_goal(username, goal_type, target_value, target_date, current_value=None):
    """Create a new health goal."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO goals(username, goal_type, target_value, current_value, start_date, target_date, status)
               VALUES(?,?,?,?,?,?,?)""",
            (username, goal_type, target_value, current_value, 
             datetime.now().strftime("%Y-%m-%d"), target_date, 'active')
        )


def update_goal_progress(goal_id, current_value, achieved=False):
    """Update goal progress."""
    with get_conn() as conn:
        if achieved:
            conn.execute(
                "UPDATE goals SET current_value=?, status='achieved', achieved_date=? WHERE id=?",
                (current_value, datetime.now().strftime("%Y-%m-%d"), goal_id)
            )
        else:
            conn.execute(
                "UPDATE goals SET current_value=? WHERE id=?",
                (current_value, goal_id)
            )


def delete_goal(goal_id):
    """Delete a goal."""
    with get_conn() as conn:
        conn.execute("DELETE FROM goals WHERE id=?", (goal_id,))
