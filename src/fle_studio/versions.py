"""Version numbers (PEP 440) and the shape of a release tag.

Used by the update plugin (is there a newer release?) and by `fle-studio update` (is the service running what was
installed?). Standard library only, so it also works while the packages around it are being replaced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# A tag that may reach a command line or a link. Plain release tags only.
SAFE_TAG = re.compile(r"^v?\d+(?:\.\d+){0,3}(?:[-.]?(?:a|b|rc|alpha|beta)[-.]?\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")

_VERSION = re.compile(
    r"""^\s*v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)
    (?:[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|rc|c)[-_.]?(?P<pre_n>\d*))?
    (?:-(?P<post_implicit>\d+)|[-_.]?(?:post|rev|r)[-_.]?(?P<post_n>\d*))?
    (?:[-_.]?dev[-_.]?(?P<dev_n>\d*))?
    (?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?\s*$""",
    re.I | re.X,
)
_PHASE = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


@dataclass(frozen=True)
class Version:
    text: str  # the version as people write it: "0.2.0", "0.2.0rc1"
    key: tuple[Any, ...]
    prerelease: bool

    def __lt__(self, other: Version) -> bool:
        return self.key < other.key

    def __gt__(self, other: Version) -> bool:
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Version) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def parse_version(text: str) -> Version | None:
    """A PEP 440 version (a leading "v" is allowed, as in release tags), or None when it is not one."""
    m = _VERSION.match(text or "")
    if not m:
        return None
    release = [int(x) for x in m["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()  # 1.0 == 1.0.0
    pre = (_PHASE[m["pre_l"].lower()], int(m["pre_n"] or 0)) if m["pre_l"] else None
    has_post = m["post_implicit"] is not None or m["post_n"] is not None or "post" in (text or "").lower()
    post = int(m["post_implicit"] or m["post_n"] or 0) if has_post else None
    dev = int(m["dev_n"] or 0) if re.search(r"dev", text or "", re.I) else None
    # Same order as the packaging library: a dev release of 1.0 comes before 1.0a1, which comes before 1.0.
    if pre is None and post is None and dev is not None:
        pre_key: tuple[int, int] = (-1, 0)
    elif pre is None:
        pre_key = (3, 0)
    else:
        pre_key = pre
    post_key = (-1,) if post is None else (post,)
    dev_key = (1, 0) if dev is None else (0, dev)
    key = (int(m["epoch"] or 0), tuple(release), pre_key, post_key, dev_key)
    return Version(text.strip().lstrip("vV"), key, pre is not None or dev is not None)


def is_newer(candidate: str, current: str) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    return bool(a and b and a > b)


def same_version(a: str, b: str) -> bool:
    """Do two version texts name the same version ("v0.4.1" and "0.4.1" do)? False when either is no version."""
    va, vb = parse_version(a), parse_version(b)
    return bool(va and vb and va == vb)
