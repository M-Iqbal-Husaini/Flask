from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)

from werkzeug.security import generate_password_hash, check_password_hash

import sqlite3
import os
import re
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ============================================================
# BASIC APP & CONFIG
# ============================================================

app = Flask(__name__)
CORS(app)

# Ganti SECRET KEY pada production
app.config["JWT_SECRET_KEY"] = "super-secret-sawit-key"
jwt = JWTManager(app)

DB_NAME = "sawit.db"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DB HELPER
# ============================================================

def conn():
    c = sqlite3.connect(DB_NAME)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = conn()

    # USERS
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      email TEXT UNIQUE,
      password TEXT,
      role TEXT DEFAULT 'user'
    )
    """)

    # LAHAN user
    c.execute("""
    CREATE TABLE IF NOT EXISTS lahan(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      nama TEXT,
      umur_tahun REAL,
      frekuensi_tahun REAL,
      ph_tanah_normal REAL,
      pemupukan_terakhir TEXT,   -- 'YYYY-MM-DD'
      FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # PREDIKSI LAHAN (riwayat prediksi user)
    c.execute("""
    CREATE TABLE IF NOT EXISTS prediksi_lahan(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      lahan_id INTEGER,
      selisih_hari REAL,
      tanggal_pemupukan_berikutnya TEXT,
      created_at TEXT,
      FOREIGN KEY(lahan_id) REFERENCES lahan(id)
    )
    """)

    # DATASET (admin)
    c.execute("""
    CREATE TABLE IF NOT EXISTS dataset(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      note TEXT,
      created_at TEXT
    )
    """)

    # ITEM DATASET (admin)
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
    )
    """)

    # RIWAYAT TRAINING MODEL
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_history(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset_id INTEGER,
      mae REAL,
      r2 REAL,
      trained_at TEXT,
      FOREIGN KEY(dataset_id) REFERENCES dataset(id)
    )
    """)

    # Seed admin default kalau belum ada
    admin_email = "admin@sawit.com"
    row = c.execute(
        "SELECT id FROM users WHERE email=?", (admin_email,)
    ).fetchone()
    if not row:
        c.execute(
            """
            INSERT INTO users(name, email, password, role)
            VALUES (?,?,?,?)
            """,
            (
                "Admin",
                admin_email,
                generate_password_hash("admin123"),
                "admin",
            ),
        )

    c.commit()
    c.close()


init_db()


# ============================================================
# HELPER LAIN
# ============================================================

def to_float(value):
    """Coba konversi ke float, handle string '3 tahun', dsb."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+", value)
        if m:
            return float(m.group())
    raise ValueError(f"tidak bisa konversi ke angka: {value!r}")


def require_admin():
    claims = get_jwt()
    return claims.get("role") == "admin"


# ============================================================
# AUTH: REGISTER & LOGIN
# ============================================================

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({"error": "name, email, dan password wajib diisi"}), 400

    c = conn()
    exists = c.execute(
        "SELECT id FROM users WHERE email=?", (email,)
    ).fetchone()
    if exists:
        c.close()
        return jsonify({"error": "Email sudah terdaftar"}), 400

    hash_pw = generate_password_hash(password)
    c.execute(
        """
        INSERT INTO users(name, email, password, role)
        VALUES (?,?,?, 'user')
        """,
        (name, email, hash_pw),
    )
    c.commit()
    c.close()

    return jsonify({"message": "Registrasi berhasil"}), 200


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "email dan password wajib diisi"}), 400

    c = conn()
    user = c.execute(
        "SELECT * FROM users WHERE email=?", (email,)
    ).fetchone()
    c.close()

    if not user:
        return jsonify({"error": "Email tidak ditemukan"}), 401

    if not check_password_hash(user["password"], password):
        return jsonify({"error": "Password salah"}), 401

    # identity = user_id, klaim tambahan berisi role, name, email
        # identity = string user_id, klaim tambahan berisi role, name, email
    claims = {
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }

    user_id_str = str(user["id"])      # <-- PENTING: jadikan string

    access_token = create_access_token(
        identity=user_id_str,
        additional_claims=claims,
    )

    return jsonify(
        {
            "access_token": access_token,
            "role": user["role"],
            "name": user["name"],
            "email": user["email"],
        }
    ), 200


# ============================================================
# ENDPOINT USER – LAHAN & PREDIKSI
# ============================================================

@app.route("/lahan", methods=["GET"])
@jwt_required()
def get_lahan():
    user_id = int(get_jwt_identity())

    c = conn()
    r = c.execute(
        "SELECT * FROM lahan WHERE user_id=? ORDER BY id DESC",
        (user_id,),
    )
    data = [dict(x) for x in r.fetchall()]
    c.close()
    return jsonify(data)


@app.route("/lahan", methods=["POST"])
@jwt_required()
def create_lahan():
    user_id = int(get_jwt_identity())

    d = request.get_json() or {}
    nama = d.get("nama")
    umur = d.get("umur_tahun")
    freq = d.get("frekuensi_tahun")
    ph = d.get("ph_tanah_normal")
    last = d.get("pemupukan_terakhir")  # "YYYY-MM-DD"

    if not all([nama, umur, freq, ph, last]):
        return jsonify({"error": "data lahan belum lengkap"}), 400

    c = conn()
    c.execute(
        """
        INSERT INTO lahan(user_id, nama, umur_tahun, frekuensi_tahun,
                          ph_tanah_normal, pemupukan_terakhir)
        VALUES (?,?,?,?,?,?)
        """,
        (user_id, nama, umur, freq, ph, last),
    )
    c.commit()
    c.close()
    return jsonify({"message": "Lahan tersimpan"}), 200


@app.route("/lahan/<int:lahan_id>/prediksi", methods=["GET"])
@jwt_required()
def get_prediksi_lahan(lahan_id):
    user_id = int(get_jwt_identity())

    c = conn()
    row = c.execute(
        "SELECT * FROM lahan WHERE id=? AND user_id=?", (lahan_id, user_id)
    ).fetchone()
    if not row:
        c.close()
        return jsonify({"error": "Lahan tidak ditemukan"}), 404

    r = c.execute(
        """
        SELECT * FROM prediksi_lahan
        WHERE lahan_id=?
        ORDER BY created_at DESC
        """,
        (lahan_id,),
    )
    data = [dict(x) for x in r.fetchall()]
    c.close()
    return jsonify(data)


@app.route("/lahan/<int:lahan_id>/predict", methods=["POST"])
@jwt_required()
def predict_lahan(lahan_id):
    user_id = int(get_jwt_identity())

    if not os.path.exists("model_rf.pkl"):
        return jsonify({"error": "model belum ada, minta admin training dulu"}), 400

    c = conn()
    lahan = c.execute(
        "SELECT * FROM lahan WHERE id=? AND user_id=?",
        (lahan_id, user_id),
    ).fetchone()
    if not lahan:
        c.close()
        return jsonify({"error": "Lahan tidak ditemukan"}), 404

    try:
        umur = float(lahan["umur_tahun"])
        freq = float(lahan["frekuensi_tahun"])
        ph = float(lahan["ph_tanah_normal"])
        last_date = datetime.strptime(lahan["pemupukan_terakhir"], "%Y-%m-%d")
    except Exception:
        c.close()
        return jsonify({"error": "Data lahan tidak valid"}), 400

    model = joblib.load("model_rf.pkl")
    X = [[umur, freq, ph]]
    selisih = float(model.predict(X)[0])
    hari_bulat = int(round(selisih))
    next_date = (last_date + timedelta(days=hari_bulat)).strftime("%Y-%m-%d")

    c.execute(
        """
        INSERT INTO prediksi_lahan(lahan_id, selisih_hari,
          tanggal_pemupukan_berikutnya, created_at)
        VALUES (?,?,?, datetime('now'))
        """,
        (lahan_id, selisih, next_date),
    )
    c.commit()
    c.close()

    return jsonify(
        {
            "selisih_hari": selisih,
            "hari_bulat": hari_bulat,
            "tanggal_pemupukan_berikutnya": next_date,
        }
    ), 200


# ============================================================
# ENDPOINT ADMIN – DATASET & UPLOAD
# ============================================================

@app.route("/dataset", methods=["GET"])
@jwt_required()
def get_dataset():
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    c = conn()
    r = c.execute("SELECT * FROM dataset ORDER BY id DESC")
    data = [dict(x) for x in r.fetchall()]
    c.close()
    return jsonify(data)


@app.route("/dataset/create", methods=["POST"])
@jwt_required()
def create_dataset():
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    d = request.get_json() or {}
    name = d.get("name")
    note = d.get("note", "")

    if not name:
        return jsonify({"error": "name required"}), 400

    c = conn()
    c.execute(
        """
        INSERT INTO dataset(name, note, created_at)
        VALUES (?,?, datetime('now'))
        """,
        (name, note),
    )
    c.commit()
    c.close()
    return jsonify({"message": "Dataset dibuat"}), 200


@app.route("/dataset/item/<int:dataset_id>", methods=["GET"])
@jwt_required()
def item_list(dataset_id):
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    c = conn()
    r = c.execute(
        "SELECT * FROM dataset_item WHERE dataset_id=?",
        (dataset_id,),
    )
    data = [dict(x) for x in r.fetchall()]
    c.close()
    return jsonify(data)


@app.route("/dataset/upload/<int:dataset_id>", methods=["POST"])
@jwt_required()
def upload_dataset(dataset_id):
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400

    file = request.files["file"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()  # .csv / .xlsx / .xls

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Baca file ke DataFrame
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif ext == ".csv":
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin1")
        else:
            return jsonify({"error": f"unsupported extension: {ext}"}), 400
    except Exception as e:
        return jsonify({"error": f"failed to read file: {e}"}), 400

    required_cols = [
        "Umur (tahun)",
        "Pemupukan Terakhir",
        "Frekuensi/Tahun",
        "pH tanah normal",
        "Pemupukan_berikutnya",
        "Selisih_hari",
    ]
    for col in required_cols:
        if col not in df.columns:
            return jsonify(
                {
                    "error": f"kolom '{col}' tidak ditemukan di file "
                             "(header harus persis)"
                }
            ), 400

    # Format tanggal ke string
    df["Pemupukan Terakhir"] = pd.to_datetime(
        df["Pemupukan Terakhir"]
    ).dt.strftime("%Y-%m-%d")

    df["Pemupukan_berikutnya"] = pd.to_datetime(
        df["Pemupukan_berikutnya"]
    ).dt.strftime("%Y-%m-%d")

    c = conn()
    for _, row in df.iterrows():
        try:
            umur = to_float(row["Umur (tahun)"])
            freq = to_float(row["Frekuensi/Tahun"])
            ph = to_float(row["pH tanah normal"])
            selisih = to_float(row["Selisih_hari"])
        except ValueError as e:
            c.close()
            return jsonify({"error": str(e)}), 400

        c.execute(
            """
            INSERT INTO dataset_item (
                dataset_id,
                umur_tahun,
                pemupukan_terakhir,
                frekuensi_tahun,
                ph_tanah_normal,
                pemupukan_berikutnya,
                selisih_hari
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                umur,
                row["Pemupukan Terakhir"],
                freq,
                ph,
                row["Pemupukan_berikutnya"],
                selisih,
            ),
        )
    c.commit()
    c.close()

    return jsonify({"message": "upload success"}), 200


# ============================================================
# TRAIN MODEL (ADMIN) + RIWAYAT TRAINING
# ============================================================

@app.route("/train/<int:dataset_id>", methods=["POST"])
@jwt_required()
def train(dataset_id):
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    c = conn()
    # ambil semua baris untuk dataset ini
    df_raw = pd.read_sql_query(
        "SELECT * FROM dataset_item WHERE dataset_id=?",
        c,
        params=(dataset_id,),
    )

    if df_raw.shape[0] < 5:
        c.close()
        return jsonify({"error": "minimal 5 data untuk training"}), 400

    # --------- BERSIHKAN DATA DARI NaN / TEKS ANEH ----------
    cols = ["umur_tahun", "frekuensi_tahun", "ph_tanah_normal", "selisih_hari"]
    df = df_raw[cols].copy()

    # paksa ke numerik, yang gagal jadi NaN
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # buang baris yang masih ada NaN
    sebelum = df.shape[0]
    df = df.dropna()
    sesudah = df.shape[0]

    if sesudah < 5:
        c.close()
        return jsonify({
            "error": (
                f"Data valid untuk training kurang dari 5 baris "
                f"(sebelum: {sebelum}, sesudah dibersihkan: {sesudah}). "
                "Periksa kembali isi dataset (pastikan angka tidak kosong "
                "dan formatnya benar)."
            )
        }), 400

    X = df[["umur_tahun", "frekuensi_tahun", "ph_tanah_normal"]]
    y = df["selisih_hari"]

    # --------- TRAIN MODEL ----------
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    # Simpan model global
    joblib.dump(model, "model_rf.pkl")

    # Simpan riwayat training
    c.execute(
        """
        INSERT INTO training_history(dataset_id, mae, r2, trained_at)
        VALUES (?,?,?, datetime('now'))
        """,
        (dataset_id, mae, r2),
    )
    c.commit()
    c.close()

    return jsonify({"message": "training OK", "mae": mae, "r2": r2}), 200


@app.route("/training/history", methods=["GET"])
@jwt_required()
def training_history():
    if not require_admin():
        return jsonify({"error": "Hanya admin yang boleh mengakses"}), 403

    dataset_id = request.args.get("dataset_id", type=int)

    c = conn()
    if dataset_id:
        r = c.execute(
            """
            SELECT * FROM training_history
            WHERE dataset_id=?
            ORDER BY trained_at DESC
            """,
            (dataset_id,),
        )
    else:
        r = c.execute(
            """
            SELECT * FROM training_history
            ORDER BY trained_at DESC
            """
        )
    data = [dict(x) for x in r.fetchall()]
    c.close()
    return jsonify(data)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
