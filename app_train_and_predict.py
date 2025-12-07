#!/usr/bin/env python3
"""
app_train_and_predict_twofield.py
Flask app yang mendukung input umur sebagai dua field: umur_tahun & umur_bulan.
Endpoints:
 - GET  /status        -> cek apakah model ada
 - POST /train         -> train model dari dataset (dataset_project.xlsx default)
 - POST /predict       -> prediksi, terima umur_tahun & umur_bulan atau umur_num
Model artifact: rf_pipeline.joblib
"""
import os
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
MODEL_PATH = "rf_pipeline.joblib"
DEFAULT_DATASET = "dataset_sawit.xlsx"

app = Flask(__name__)
CORS(app)

# -----------------------------
# Utilities
# -----------------------------
def years_months_to_years(y, m):
    try:
        yv = int(y) if y is not None and str(y).strip() != "" else 0
    except:
        yv = 0
    try:
        mv = int(m) if m is not None and str(m).strip() != "" else 0
    except:
        mv = 0
    return float(yv) + float(mv) / 12.0

def coerce_int(val, default=0):
    try:
        return int(val)
    except:
        return default

def load_and_prepare_dataframe(path):
    """
    Baca dataset dan hasilkan X, y, numeric_cols, cat_cols.
    Menerima dataset yang mungkin memiliki:
      - umur_tahun, umur_bulan  (preferred)
      - atau umur_num (tahun float)
      - atau umur (string) -- best effort
    Juga akan menambahkan fitur umur_bulan dan umur_num untuk konsistensi.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported dataset type: " + ext)

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # parse tanggal columns if present
    if "pemupukan_terakhir" in df.columns:
        df["pemupukan_terakhir"] = pd.to_datetime(df["pemupukan_terakhir"], errors="coerce")
    if "pemupukan_berikutnya" in df.columns:
        df["pemupukan_berikutnya"] = pd.to_datetime(df["pemupukan_berikutnya"], errors="coerce")

    # derive selisih_hari if possible
    if "selisih_hari" not in df.columns and "pemupukan_terakhir" in df.columns and "pemupukan_berikutnya" in df.columns:
        df["selisih_hari"] = (df["pemupukan_berikutnya"] - df["pemupukan_terakhir"]).dt.days

    # fallback frequency -> selisih
    if "frekuensi_num" in df.columns and "selisih_hari" in df.columns:
        mask = df["selisih_hari"].isna() & df["frekuensi_num"].notna()
        df.loc[mask, "selisih_hari"] = (365.0 / df.loc[mask, "frekuensi_num"]).round()

    if "selisih_hari" in df.columns:
        df["selisih_hari"] = pd.to_numeric(df["selisih_hari"], errors="coerce")

    if "selisih_hari" not in df.columns:
        raise ValueError("Dataset must contain 'selisih_hari' column or pemupukan dates to derive it.")

    # ---- handle umur fields ----
    # if separate umur_tahun & umur_bulan present -> create umur_num & umur_bulan canonical
    if "umur_tahun" in df.columns or "umur_bulan" in df.columns:
        ycol = "umur_tahun" if "umur_tahun" in df.columns else None
        mcol = "umur_bulan" if "umur_bulan" in df.columns else None
        # coerce numeric
        if ycol:
            df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
        if mcol:
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
        # create canonical umur_bulan and umur_num
        df["umur_bulan"] = df.get("umur_bulan", 0).fillna(0)
        if "umur_tahun" in df.columns:
            # combine
            df["umur_bulan"] = df["umur_bulan"].fillna(0) + df["umur_tahun"].fillna(0).astype(int) * 12
        df["umur_bulan"] = pd.to_numeric(df["umur_bulan"], errors="coerce").fillna(0).astype(int)
        df["umur_num"] = (df["umur_bulan"].astype(float) / 12.0).round(6)
    else:
        # try fallback: umur_num column or umur text
        if "umur_num" in df.columns:
            df["umur_num"] = pd.to_numeric(df["umur_num"], errors="coerce")
            # also create umur_bulan
            df["umur_bulan"] = (df["umur_num"] * 12.0).round().astype(int)
        elif "umur" in df.columns:
            # best-effort parse simple forms like "1 tahun 2 bulan" or "14 bulan" or "1.5"
            def parse_simple_age(v):
                if pd.isna(v): return np.nan
                s = str(v).strip().lower()
                import re
                # decimal
                m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*$", s)
                if m:
                    val = float(m.group(1).replace(",", "."))
                    return val
                my = re.search(r"(\d+)\s*(tahun|thn|y)", s)
                mm = re.search(r"(\d+)\s*(bulan|bln|m)", s)
                y = int(my.group(1)) if my else 0
                mo = int(mm.group(1)) if mm else 0
                if y==0 and mo==0:
                    single = re.match(r"^(\d+)\s*$", s)
                    if single:
                        y = int(single.group(1))
                return float(y + mo/12.0)
            df["umur_num"] = df["umur"].apply(parse_simple_age)
            df["umur_bulan"] = (df["umur_num"] * 12.0).round().astype(int)
        else:
            # no umur info found -> error (model needs umur)
            raise ValueError("Dataset must contain umur info: 'umur_num' or ('umur_tahun' and/or 'umur_bulan') or 'umur' textual.")

    # ensure numeric conversion
    df["umur_num"] = pd.to_numeric(df["umur_num"], errors="coerce")
    df["umur_bulan"] = pd.to_numeric(df["umur_bulan"], errors="coerce").fillna(0).astype(int)

    # drop rows w/o target or nonpositive
    df = df[pd.to_numeric(df["selisih_hari"], errors="coerce").notna()]
    df["selisih_hari"] = df["selisih_hari"].astype(float)
    df = df[df["selisih_hari"] > 0]

    # date features
    if "pemupukan_terakhir" in df.columns:
        df["bulan_terakhir"] = df["pemupukan_terakhir"].dt.month
        df["hari_ke_dalam_tahun"] = df["pemupukan_terakhir"].dt.dayofyear
        df["hari_sejak_terakhir"] = (pd.Timestamp.today().normalize() - df["pemupukan_terakhir"]).dt.days

    # feature lists: include umur_num (float years) and umur_bulan (int months)
    numeric_candidates = ["umur_num", "umur_bulan", "frekuensi_num", "ph_tanah_normal", "bulan_terakhir", "hari_ke_dalam_tahun", "hari_sejak_terakhir"]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = [c for c in ["nama_petani"] if c in df.columns]

    FEATURES = numeric_cols + cat_cols
    TARGET = "selisih_hari"

    if len(FEATURES) == 0:
        raise ValueError("No features found. Check dataset columns.")

    X = df[FEATURES].copy()
    y = df[TARGET].copy().astype(float)

    # drop rows where all features NaN
    X = X.dropna(how="all")
    if len(X) == 0:
        raise ValueError("No usable rows after cleaning.")

    return X, y, numeric_cols, cat_cols

# -----------------------------
# Endpoints
# -----------------------------
@app.route("/status", methods=["GET"])
def status():
    return jsonify({"model_exists": os.path.exists(MODEL_PATH), "model_path": MODEL_PATH}), 200

@app.route("/train", methods=["POST"])
def train():
    body = request.get_json() or {}
    dataset_path = body.get("dataset_path", DEFAULT_DATASET)
    n_iter = int(body.get("n_iter", 6))
    test_size = float(body.get("test_size", 0.2))

    if not os.path.exists(dataset_path):
        return jsonify({"error": "dataset not found", "dataset_path": dataset_path}), 400

    start = time.time()
    try:
        X, y, numeric_cols, cat_cols = load_and_prepare_dataframe(dataset_path)

        # preprocessors
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        baseline = Pipeline([("preproc", preprocessor), ("model", DummyRegressor(strategy="median"))])
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)

        pipeline = Pipeline([("preproc", preprocessor), ("model", RandomForestRegressor(random_state=42))])

        param_dist = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
        cv = KFold(n_splits=4, shuffle=True, random_state=42)
        search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter,
                                    scoring="neg_mean_absolute_error", cv=cv, random_state=42, n_jobs=1, verbose=1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        joblib.dump(best_model, MODEL_PATH)
        elapsed = time.time() - start
        return jsonify({"status": "trained", "elapsed_seconds": elapsed, "baseline_mae": baseline_mae,
                        "mae": mae, "rmse": rmse, "r2": r2, "best_params": search.best_params_}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal", "detail": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with either:
      - umur_tahun (int) and umur_bulan (int)
    OR
      - umur_num (float) OR umur_bulan (int)
    And also:
      - nama_petani, ph_tanah_normal, frekuensi_num, tanggal_terakhir (YYYY-MM-DD)
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "model not trained yet; call /train first"}), 400

    try:
        pipeline = joblib.load(MODEL_PATH)
        data = request.get_json() or {}

        required = ["nama_petani", "ph_tanah_normal", "frekuensi_num", "tanggal_terakhir"]
        for r in required:
            if r not in data:
                return jsonify({"error": f"missing {r}"}), 400

        # compute umur fields
        if "umur_tahun" in data or "umur_bulan" in data:
            ut = data.get("umur_tahun", 0)
            ub = data.get("umur_bulan", 0)
            umur_num = years_months_to_years(ut, ub)
            umur_bulan = int(round(float(ub) + (int(ut) if ut not in (None, "") else 0) * 12))
        elif "umur_num" in data:
            umur_num = float(data.get("umur_num"))
            umur_bulan = int(round(umur_num * 12.0))
        elif "umur_bulan" in data:
            umur_bulan = int(data.get("umur_bulan"))
            umur_num = float(umur_bulan) / 12.0
        else:
            return jsonify({"error": "missing umur info; provide umur_tahun &/atau umur_bulan or umur_num"}), 400

        tanggal_terakhir = pd.to_datetime(data["tanggal_terakhir"], errors="coerce")
        if pd.isna(tanggal_terakhir):
            return jsonify({"error": "invalid tanggal_terakhir format"}), 400

        row = {
            "nama_petani": data["nama_petani"],
            "umur_num": float(umur_num),
            "umur_bulan": int(umur_bulan),
            "ph_tanah_normal": float(data["ph_tanah_normal"]),
            "frekuensi_num": float(data["frekuensi_num"]),
            "bulan_terakhir": int(tanggal_terakhir.month),
            "hari_ke_dalam_tahun": int(tanggal_terakhir.dayofyear),
            "hari_sejak_terakhir": int((pd.Timestamp.today().normalize() - tanggal_terakhir).days)
        }

        df = pd.DataFrame([row])
        pred_days = pipeline.predict(df)[0]
        pred_days_int = int(round(pred_days))
        pred_date = tanggal_terakhir + pd.Timedelta(days=pred_days_int)

        return jsonify({"pred_days": float(pred_days), "pred_days_int": pred_days_int, "pred_date": str(pred_date.date())}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
