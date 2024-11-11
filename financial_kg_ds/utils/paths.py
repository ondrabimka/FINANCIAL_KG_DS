from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "financial_kg_ds" / "data"
HISTORICAL_DATA_FILE = DATA_DIR / "historical_prices"