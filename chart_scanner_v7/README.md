# Chart Scanner V7

Chart Scanner is a local Streamlit stock-pattern scanner. It scans a broad stock universe or a custom ticker list and returns pattern candidates with score, direction, entry, stop, target, setup status, distance to entry, and risk/reward.

## Quick start on Windows

1. Unzip this folder.
2. Double-click `Run_Chart_Scanner.bat`.
3. The first run may take a few minutes because it creates a Python environment and installs packages.
4. Keep the black command window open while using the scanner.
5. The app should open automatically in your browser, usually at `http://localhost:8501`.

## Manual start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Uses Yahoo/yfinance data for prototyping, which can be delayed or rate-limited.
- For professional real-time scanning, connect to a licensed data provider such as Polygon, Alpaca, Tradier, or Interactive Brokers.
- Always verify the chart visually before entering a trade.
