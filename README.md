# Predictive Maintenance Prototype

A predictive maintenance dashboard and API that uses a NASA C-MAPSS FD001-inspired dataset to:

- detect anomalies using an autoencoder reconstruction error
- predict remaining useful life (RUL) using a Random Forest or XGBoost model
- visualize engine health over time in a Streamlit dashboard

## Architecture

- Backend API: `pdm/api/index.py`
- Dashboard: `pdm/streamlit_app.py`
- Model training: `pdm/train.py`
- Trained artifacts: `pdm/models/`

## Quick start

```bash
cd predictive_maintenance_prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r pdm/requirements.txt -r pdm/requirements_app.txt

# Start backend
cd pdm
uvicorn api.index:app --host 0.0.0.0 --port 8000

# In another terminal
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

- http://localhost:8501 for the dashboard
- http://localhost:8000/docs for the API docs

## Notes

The app is configured to default to the local FastAPI backend at `http://localhost:8000`.

## Data and models

This prototype uses the NASA C-MAPSS TD/FD001 dataset pattern and ships with trained model artifacts in `pdm/models`.
