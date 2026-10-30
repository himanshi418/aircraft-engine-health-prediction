# Aircraft Engine Health Prediction — Streamlit

A dark-mode Streamlit dashboard for an educational Predictive Analytics project that estimates Remaining Useful Life (RUL) from three features: `time_in_cycles`, `sensor_2`, and `sensor_3`.

## Run locally (Mac / Windows)

1. Extract this ZIP and open the `aircraft_engine_streamlit` folder in VS Code.
2. Open **Terminal → New Terminal** in VS Code.
3. Create and activate an environment (recommended):

   **Mac:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   **Windows:**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

The `.streamlit/config.toml` file sets the global dark theme. Keep the `.streamlit` folder beside `app.py` and run Streamlit from this project folder.

## Use the NASA dataset for your final project

This ZIP includes a **synthetic demo model** only so the app opens immediately for UI testing. It is explicitly labelled in the app and its demo metrics are not shown as validation results. Do not use its predictions or metrics in your report.

For the actual project model:
1. Put the NASA C-MAPSS FD001 `train_FD001.csv` file in this folder.
2. Run:
   ```bash
   python3 train_model.py
   ```
   (On Windows, `python train_model.py` also works.)
3. Restart the app. It will train/save a three-feature model from the dataset.

The live dashboard's three-feature model is a simplified demo and is **not the same** as the 18-feature models in `aircraft_rul_analysis.ipynb`. Keep their metrics separate in your presentation. If you already have a trained, correct three-feature model, replace `model/rul_rf_3feat.joblib` with that model bundle rather than retraining.

## Deploy on Streamlit Community Cloud

1. Upload this project folder to a GitHub repository.
2. Ensure `app.py`, `requirements.txt`, `.streamlit/config.toml`, `train_model.py`, and `model/rul_rf_3feat.joblib` are committed.
3. In Streamlit Community Cloud, choose **Create app**, select your repository, branch, and `app.py`, then deploy.
4. If you change the theme, commit the config file and reboot/redeploy the app.

## Important limitation

NASA C-MAPSS is simulated turbofan degradation data. This is a college/educational project, not a certified aircraft-maintenance tool and not suitable for real flight-safety decisions.
