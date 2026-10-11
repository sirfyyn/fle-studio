import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fake_llama_tools  # noqa: E402

fake_llama_tools.main("perplexity")
