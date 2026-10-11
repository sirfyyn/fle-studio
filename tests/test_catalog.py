"""The catalog has a second mixture-of-experts family, and every entry can be downloaded and shown."""

from __future__ import annotations

from fle_studio import catalog


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
