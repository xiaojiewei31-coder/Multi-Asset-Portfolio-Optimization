from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

from portfolio_optimization.pipeline import main


if __name__ == "__main__":
    main()
