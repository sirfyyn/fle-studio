from __future__ import annotations

import json

import pytest

from fle_studio_forge import recipe as recipes


def test_a_new_recipe_has_the_defaults_and_the_schema():
    r = recipes.new("My Quant!")
    assert r["schema"] == 1 and r["kind"] == "fle-forge-recipe" and r["name"] == "My-Quant"
    assert r["mode"] == "hybrid" and r["verify"]["enabled"] and not r["reap"]["enabled"]
    assert r["created"].endswith("Z") and r["studio"]


def test_unknown_keys_are_dropped_and_values_are_checked():
    r = recipes.normalize({"name": "x", "surprise": 1, "hybrid": {"base": "B", "run": "rm -rf /"}})
    assert "surprise" not in r and "run" not in r["hybrid"]
    for bad, message in [
        ({"mode": "magic"}, "mode"),
        ({"schema": 2}, "schema 2"),
        ({"kind": "other"}, "not a Forge recipe"),
        ({"reap": {"enabled": True, "keep": 0}}, "reap.keep"),
        ({"reap": {"score": "luck"}}, "reap.score"),
        ({"quantize": {"type": "q99"}}, "unknown type"),
        ({"quantize": {"rules": [{"pattern": "a b; rm", "type": "q4_0"}]}}, "bad pattern"),
        ({"quantize": {"rules": [{"pattern": "exps", "type": "nope"}]}}, "unknown type"),
        ({"hybrid": {"rules": [{"pattern": "$(x)", "source": "K"}]}}, "bad pattern"),
        ({"hybrid": {"layers": {"x": "K"}}}, "not a layer number"),
        ({"hybrid": {"objective": "vibes"}}, "objective"),
        ({"imatrix": {"chunks": "many"}}, "number"),
        ({"imatrix": {"context": 3}}, "between"),
        ({"sources": {"1bad": {}}}, "source id"),
        ({"sources": {"B": {"role": "boss"}}}, "role"),
        ({"sources": {"B": {"sha256": "xyz"}}}, "sha256"),
    ]:
        with pytest.raises(recipes.RecipeError, match=message):
            recipes.normalize(bad)
    with pytest.raises(recipes.RecipeError):
        recipes.normalize([])  # type: ignore[arg-type]


def test_the_file_name_follows_the_path_and_types_are_lowercased():
    r = recipes.normalize(
        {"sources": {"B": {"path": "C:\\models\\big\\base-Q8_0.gguf"}}, "quantize": {"type": "IQ3_S"}}
    )
    assert r["sources"]["B"]["file"] == "base-Q8_0.gguf" and r["quantize"]["type"] == "iq3_s"


def test_layer_lists():
    assert recipes.parse_layer_set("0-2, 5,7-8") == {0, 1, 2, 5, 7, 8}
    assert recipes.parse_layer_set("") == set()
    for bad in ("a", "3-1", "-2", "1-99999"):
        with pytest.raises(recipes.RecipeError):
            recipes.parse_layer_set(bad)


def test_sources_are_validated_against_the_mode():
    r = recipes.normalize(
        {
            "mode": "hybrid",
            "sources": {"B": {"path": "/x/b.gguf", "role": "base"}},
            "hybrid": {"base": "B", "layers": {"3": "K"}},
            "verify": {"enabled": False},
        }
    )
    assert recipes.validate_sources(r) == ["layer 3 comes from 'K', which the recipe does not list"]
    q = recipes.normalize({"mode": "quantize", "sources": {"S": {}}, "quantize": {"source": "S"}})
    problems = recipes.validate_sources(q)
    assert any("no file on this computer" in p for p in problems) and any("verify.reference" in p for p in problems)


def test_a_shared_recipe_carries_no_folders_and_binds_to_new_ones():
    r = recipes.normalize(
        {
            "name": "share",
            "sources": {"B": {"path": "/home/me/models/b.gguf", "role": "base"}},
            "imatrix": {"corpus": "/home/me/texts/corpus.txt", "file": "C:\\data\\im.gguf"},
            "reap": {"table": "/home/me/t.tsv"},
            "verify": {"text": "/home/me/wiki.txt"},
        }
    )
    shared = recipes.portable(r)
    text = recipes.dumps(shared)
    assert "/home/me" not in text and "C:" not in text
    assert shared["sources"]["B"]["path"] == "" and shared["sources"]["B"]["file"] == "b.gguf"
    assert shared["imatrix"]["corpus"] == "corpus.txt" and shared["imatrix"]["file"] == "im.gguf"
    back = recipes.loads(text)
    assert back == shared
    bound = recipes.bind(back, {"B": "/new/place/b.gguf", "Z": "/ignored"})
    assert bound["sources"]["B"]["path"] == "/new/place/b.gguf" and "Z" not in bound["sources"]
    assert r["sources"]["B"]["path"] == "/home/me/models/b.gguf"  # the original is not touched


def test_normalizing_twice_changes_nothing():
    once = recipes.normalize({"name": "a b", "hybrid": {"layers": {"2": "K"}, "errors": {"K": {"2": 0.5}}}})
    assert recipes.normalize(once) == once
    assert json.loads(recipes.dumps(once)) == once


def test_broken_json_is_reported():
    with pytest.raises(recipes.RecipeError, match="not valid JSON"):
        recipes.loads("{nope")


def test_named_mixtures_are_a_default_type_but_not_a_tensor_type():
    assert recipes.normalize({"quantize": {"type": "Q3_K_M"}})["quantize"]["type"] == "q3_k_m"
    rule = {"rules": [{"pattern": "x", "type": "q3_k_m"}]}
    for bad in ({"output_type": "q3_k_m"}, {"embedding_type": "q4_k_m"}, rule):
        with pytest.raises(recipes.RecipeError, match="unknown type"):
            recipes.normalize({"quantize": {"type": "q4_0", **bad}})
