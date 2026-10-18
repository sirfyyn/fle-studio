import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "src")
)  # fle_studio ohne installiertes Paket (Tests laufen auch aus dem Quellbaum)

import fake_llama_tools  # noqa: E402

fake_llama_tools.main("fle-reap")
