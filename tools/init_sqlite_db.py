import sqlite3

DB_PATH = "../weather.db"  # Passe ggf. an

schema = """
CREATE TABLE IF NOT EXISTS image_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    image_path TEXT NOT NULL,
    roi_image_path TEXT,
    wbg INTEGER,
    moon_phase REAL,
    moon_zenith_angle REAL,
    threshold INTEGER,
    analysis_version TEXT,
    classification INTEGER,
    comment TEXT
);

CREATE TABLE IF NOT EXISTS k_coefficients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    k1 REAL,
    k2 REAL,
    k3 REAL,
    k4 REAL,
    k5 REAL,
    k6 REAL,
    k7 REAL
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    level TEXT,
    message TEXT
);
"""

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.executescript(schema)
conn.commit()
conn.close()
print("SQLite-Tabellen wurden angelegt (falls nicht vorhanden).")
