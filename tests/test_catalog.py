"""The catalog has a second mixture-of-experts family, and every entry can be downloaded and shown."""

from __future__ import annotations

from pathlib import Path

from fle_studio import catalog
from fle_studio.server import Studio


def test_gpt_oss_is_listed_with_hub_sizes_and_an_open_license() -> None:
    small, big = catalog.BY_ID["gpt-oss-20b"], catalog.BY_ID["gpt-oss-120b"]
    for m in (small, big):
        assert m.family == "gpt-oss" and m.license == "Apache-2.0" and m.mtp is False
        assert m.kld is None and not m.measured  # nothing is claimed that was not measured
        assert m.repo.startswith("ggml-org/") and m.files[0].name.endswith("-MXFP4.gguf")
    assert small.size == 12109566624 and big.size == 63387346208
    assert len({m.family for m in catalog.CATALOG}) == 2


def test_every_entry_serialises_and_has_unique_ids_and_files() -> None:
    ids = [m.id for m in catalog.CATALOG]
    assert len(ids) == len(set(ids))
    assert len(catalog.BY_FILENAME) == sum(len(m.files) for m in catalog.CATALOG)
    for m in catalog.CATALOG:
        public = m.public()
        assert public["size"] == m.size > 0 and public["files"][0]["name"]


def test_mtp_status_for_the_badge() -> None:
    assert catalog.mtp_status(False, True, True)["state"] == "none"
    assert catalog.mtp_status(True, True, True) == {"state": "ready", "label": "MTP ready", "head": True, "vocab": True}
    half = catalog.mtp_status(True, True, False)
    assert half["state"] == "missing" and half["label"] == "MTP files missing" and half["vocab"] is False


def test_catalog_view_carries_the_mtp_status(tmp_path: Path) -> None:
    st = Studio(tmp_path / "home")
    try:
        view = {m["id"]: m["mtp_status"] for m in st.catalog_view()}
        with_mtp = next(m.id for m in catalog.CATALOG if m.mtp)
        without = next(m.id for m in catalog.CATALOG if not m.mtp)
        assert view[with_mtp]["state"] == "missing" and view[without]["state"] == "none"
        root = st.config.models_dir
        for ref in (catalog.MTP_HEAD, catalog.MTP_VOCAB):  # the sidecars appear in the models folder
            dest = root / ref.repo / ref.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
        view = {m["id"]: m["mtp_status"] for m in st.catalog_view()}
        assert view[with_mtp]["state"] == "ready" and view[without]["state"] == "none"
    finally:
        st.engine.shutdown()
