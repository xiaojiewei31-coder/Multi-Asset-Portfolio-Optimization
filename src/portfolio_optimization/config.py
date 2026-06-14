from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

RISK_FREE_RATE = 0.0176

ASSET_RANGES = {
    "000012": ("O", 6, 125),
    "000852": ("I", 6, 125),
    "000905": ("F", 6, 125),
    "399300": ("C", 6, 125),
    "H11001": ("L", 6, 125),
}

ASSET_LABELS = {
    "000012": "SSE Government Bond",
    "000852": "CSI 1000",
    "000905": "CSI 500",
    "399300": "CSI 300",
    "H11001": "CSI Aggregate Bond",
}

BASE_ASSETS = ["000012", "000852", "000905", "399300", "H11001"]
ALTERNATIVE_ASSETS = ["HSREIT", "GOLD", "HSI", "SPX"]
NINE_ASSETS = ALTERNATIVE_ASSETS + BASE_ASSETS

ALTERNATIVE_PRICE_RANGES = {
    "HSREIT": ("A", "B", 5, 136),
    "GOLD": ("D", "E", 5, 136),
    "HSI": ("G", "H", 5, 136),
    "SPX": ("J", "K", 5, 136),
}

ASSET_LABELS.update(
    {
        "HSREIT": "Hang Seng REIT",
        "GOLD": "Gold ETF",
        "HSI": "Hang Seng Index",
        "SPX": "S&P 500",
    }
)
