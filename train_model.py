"""Train the 3-feature demonstration model.

If train_FD001.csv is in this folder, use the NASA C-MAPSS FD001 training file.
Otherwise create a clearly labelled synthetic demo model so the app can run for UI testing.
The synthetic model is NOT a scientific result and must not be reported as model performance.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["time_in_cycles", "sensor_2", "sensor_3"]
COLS = ["unit_number", "time_in_cycles"] + [f"setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
train_file = ROOT / "train_FD001.csv"

if train_file.exists():
    df = pd.read_csv(train_file)
    if "unit_number" not in df.columns:
        df = pd.read_csv(train_file, sep=r"\s+", header=None)
        df = df.iloc[:, :26]
        df.columns = COLS[:df.shape[1]]
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycle - df["time_in_cycles"]
    X = df[FEATURES].astype(float)
    y = df["RUL"].astype(float)
    source = "NASA C-MAPSS FD001 training file"
    train_rows = len(df)
    n_engines = int(df["unit_number"].nunique())
else:
    # Synthetic values solely to let the UI run before the real dataset is added.
    rng = np.random.default_rng(42)
    n = 5000
    cycles = rng.integers(1, 363, n)
    s2 = np.clip(rng.normal(642, 0.7, n) + cycles * 0.0015, 640, 645)
    s3 = np.clip(rng.normal(1580, 3.0, n) - cycles * 0.018, 1570, 1595)
    y = np.clip(360 - cycles + (s3 - 1580) * 1.5 - (s2 - 642) * 2 + rng.normal(0, 8, n), 0, 360)
    X = pd.DataFrame({"time_in_cycles": cycles, "sensor_2": s2, "sensor_3": s3})
    source = "SYNTHETIC UI DEMO ONLY — replace by adding train_FD001.csv and rerunning"
    train_rows, n_engines = n, None

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_val)
metrics = {"MAE": float(mean_absolute_error(y_val, pred)), "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))), "R2": float(r2_score(y_val, pred))}
ranges = {c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())} for c in FEATURES}
bundle = {"model": model, "features": FEATURES, "ranges": ranges, "max_rul": float(max(y)), "train_rows": int(train_rows), "n_engines": n_engines, "validation_metrics": metrics, "validation_note": "random held-out split", "source": source}
joblib.dump(bundle, MODEL_DIR / "rul_rf_3feat.joblib")
print(f"Saved model: {MODEL_DIR / 'rul_rf_3feat.joblib'}")
print(f"Training source: {source}")
print(f"Validation: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
