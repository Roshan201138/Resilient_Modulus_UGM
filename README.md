# APP for resilient modulus prediction

Streamlit application for resilient modulus prediction using trained machine-learning models and calibrated empirical models.

Developed by: Mohammad Jawed Roshan

## Repository contents

```text
app.py                     Streamlit application
requirements.txt           Python package requirements
.python-version            Local Python version hint
.streamlit/config.toml     Streamlit visual theme
models/                    Trained ML models used by the app
assets/logos/              Optional logo images displayed at the top of the app
```

## Models included

The app loads trained `.joblib` files from the `models/` folder and ignores stacked or hybrid models.

Included model display names:

- KNN
- SVM
- RF
- DT
- LightGBM
- XGBoost
- ANN

## Required Python version

Use Python 3.10. The trained models were serialized with package versions compatible with Python 3.10 and scikit-learn 1.2.2.

## Local setup

Windows CMD or PowerShell:

```bat
py -3.10 -m venv venv310
venv310\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run app.py
```

macOS/Linux:

```bash
python3.10 -m venv venv310
source venv310/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Upload this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app from the GitHub repository.
3. Select `app.py` as the app entry point.
4. In Advanced settings, select Python 3.10.
5. Add this environment variable in the app settings if the ANN model is used:

```text
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

6. Deploy the app.

## Input columns for ML prediction

The batch prediction file should contain these columns:

- `Confining_pressure` in kPa
- `Loading_frequency` in Hz
- `Loading_cycle` in cycles
- `Deviator_stress` in kPa
- `Mean_stress` in kPa
- `Dissipated_energy` in J/m³
- `Elastic_energy` in J/m³

The app accepts common variations in column names and standardizes them internally.

## Empirical models

The empirical page calibrates coefficients automatically from a calibration dataset containing confining pressure, deviator stress, and measured resilient modulus. It implements:

- MEPDG model
- K-θ model
- Uzan model

After calibration, single empirical prediction can be performed using the calibrated coefficients.
