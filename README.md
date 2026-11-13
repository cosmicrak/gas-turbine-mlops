# Gas Turbine ML Dashboard

An academic MLOps-style deployment of the Gas Turbine CO and NOx emissions project.

## What is included

- `notebooks/`: the original Google Colab notebook used for model development.
- `model/gas_turbine_artifacts.joblib`: compressed, exported model bundle and supporting results.
- `app.py`: interactive Streamlit dashboard with prediction inputs, evaluation charts, feature importance, and optimization results.
- `Dockerfile`: container image definition for local Docker and Render deployment.

## Model details

- Model: Random Forest Regressor (multi-output)
- Inputs: AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP
- Outputs: TEY, CO, NOX
- The model was trained in Python 3.13.18 with scikit-learn 1.6.1, NumPy 2.4.6, pandas 2.2.3, and joblib 1.6.0. The dependency file pins these model-related versions for deserialization compatibility.

The model is based on historical sensor data. It is not connected to a live turbine or a safety-critical control system. The held-out test R² for NOX is negative, so NOX predictions must be treated as unreliable. The optimization result is exploratory and flagged as outside the historical operating distribution.

## Run locally without Docker

Use Python 3.13. Then, from this project directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Build and run with Docker

```bash
docker build -t gas-turbine-mlops .
docker run --rm -p 8501:8501 gas-turbine-mlops
```

Open `http://localhost:8501`.

## Deploy on Render

1. Push this project to a GitHub repository.
2. In Render, create **New → Web Service** and connect the repository.
3. Select **Docker** as the runtime. Keep the Dockerfile path as `./Dockerfile`.
4. Create the service. The Dockerfile starts Streamlit and uses Render's `PORT` environment variable.

## Artifact note

The model bundle is compressed with joblib compression level 3. It was reloaded and checked against the original artifact; the example prediction was unchanged.

## Important notebook synchronization step

The uploaded notebook copy contains older saved Differential Evolution output than the artifact just exported from the latest Colab run. Before pushing to GitHub, download the freshly executed notebook from Colab after `Runtime > Run all` and replace the file in `notebooks/`. The model evaluation metrics and documented sample prediction match; the optimizer's selected candidate values changed between runs.
