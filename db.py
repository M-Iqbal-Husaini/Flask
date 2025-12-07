import sqlite3

DB_NAME="sawit.db"

def conn():
    c=sqlite3.connect(DB_NAME)
    c.row_factory=sqlite3.Row
    return c

def init():
    c=conn()
    c.execute("""
    CREATE TABLE IF NOT EXISTS dataset(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        note TEXT,
        created_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS dataset_item(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        umur_tahun REAL,
        pemupukan_terakhir TEXT,
        frekuensi_tahun REAL,
        ph_tanah_normal REAL,
        pemupukan_berikutnya TEXT,
        selisih_hari REAL,
        FOREIGN KEY(dataset_id) REFERENCES dataset(id)
    );
    """)
    c.commit()
    c.close()
