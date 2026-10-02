"""Launch THIS checkout's scorer even when REINVENT uses another working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navgen.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "score-stdin")
    raise SystemExit(main())
