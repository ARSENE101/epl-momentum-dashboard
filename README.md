# Does early momentum predict a Premier League top-5 finish?

An open, work-in-progress analysis. A simple win/draw/loss momentum signal is tested against the table at
matchdays 15, 20 and 30. Trained on 2000/01-2024/25, tested blind on 2025/26. A live test on 2026/27 is planned.

**Signal, not proof.** One blind season is a small sample. Not betting advice.

## Run locally
    pip install -r requirements.txt
    python app.py

## Recompute the model results (only when data changes)
    pip install -r requirements-dev.txt
    python compute_results.py

## Data
- 2000/01-2024/25: football-data.co.uk (2003/04, 2004/05 and 2024/25 are incomplete)
- 2025/26: openfootball/england
