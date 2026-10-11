"""A stand-in for `uv` and for the `fle-studio` program, for the update tests.

    python fake_uv.py tool install --python 3.12 --force git+https://...@v0.2.0
    python fake_uv.py --launcher --version

Environment:
  FAKE_UV_OUT      file that gets the arguments, one JSON list per call (appended)
  FAKE_UV_EXIT     exit code of `tool install` (default 0); with a non-zero code a line is printed to stderr
  FAKE_UV_VERSION  what `--launcher --version` answers (default 9.9.9)
"""

from __future__ import annotations

import json
import os
import sys


def main(argv: list[str]) -> int:
    if argv[:1] == ["--launcher"]:
        print(f"FLE Studio {os.environ.get('FAKE_UV_VERSION', '9.9.9')}")
        return 0
    out = os.environ.get("FAKE_UV_OUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    code = int(os.environ.get("FAKE_UV_EXIT", "0"))
    if code:
        print("error: failed to fetch the release (this is the stand-in)", file=sys.stderr)
        return code
    print("Resolved 1 package (stand-in)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
