from __future__ import annotations

import json
from pathlib import Path

import pytest

from fle_studio import chat_tree, chats


def branching_chat() -> dict:
    """u1 -> a1 -> u2 (edited as u2b) -> a2 / a2b; the shown branch ends at a2b."""
    return {
        "id": "0123456789abcdef",
        "title": "Branches",
        "model": "m",
        "system": "Be brief.",
        "leaf": "a2b",
        "messages": [
            {"id": "u1", "parent": None, "role": "user", "content": "Hello"},
            {"id": "a1", "parent": "u1", "role": "assistant", "content": "Hi there"},
            {"id": "u2", "parent": "a1", "role": "user", "content": "First wording"},
            {"id": "a2", "parent": "u2", "role": "assistant", "content": "Answer one"},
            {"id": "u2b", "parent": "a1", "role": "user", "content": "Second wording"},
            {"id": "a2b", "parent": "u2b", "role": "assistant", "content": "Answer two"},
        ],
    }


def test_legacy_list_becomes_a_chain() -> None:
    old = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    nodes, leaf = chat_tree.normalize(old)
    assert [m["parent"] for m in nodes] == [None, nodes[0]["id"], nodes[1]["id"]]
    assert leaf == nodes[-1]["id"]
    assert [m["content"] for m in chat_tree.active_path(nodes, leaf)] == ["a", "b", "c"]


def test_normalize_repairs_ids_parents_and_cycles() -> None:
    nodes, leaf = chat_tree.normalize(
        [
            {"id": "x", "parent": "y", "role": "user", "content": "1"},
            {"id": "y", "parent": "x", "role": "assistant", "content": "2"},
            {"id": "x", "parent": "nope", "role": "user", "content": "3"},
            "junk",
        ],
        "missing",
    )
    ids = [m["id"] for m in nodes]
    assert len(set(ids)) == 3
    assert leaf == ids[-1]
    assert len(chat_tree.active_path(nodes, leaf)) <= 3  # terminates despite the cycle
    assert {m["parent"] for m in nodes} <= set(ids) | {None}


def test_active_path_follows_the_leaf() -> None:
    chat = branching_chat()
    assert [m["id"] for m in chat_tree.active_path(chat["messages"], "a2b")] == ["u1", "a1", "u2b", "a2b"]
    assert [m["id"] for m in chat_tree.active_path(chat["messages"], "a2")] == ["u1", "a1", "u2", "a2"]
    assert chat_tree.active_path(chat["messages"], None) == []


def test_search_text_covers_all_branches_and_attachments() -> None:
    chat = branching_chat()
    chat["messages"][0]["attachments"] = [{"kind": "text", "name": "notes.txt", "text": "needle in a haystack"}]
    text = chat_tree.search_text(chat).casefold()
    assert "first wording" in text and "second wording" in text
    assert "notes.txt" in text and "needle" in text
    assert "…" in chat_tree.snippet("x " * 200 + "needle" + " y" * 200, "NEEDLE")


def test_markdown_export_shows_only_the_active_branch() -> None:
    md = chat_tree.export_markdown(branching_chat(), "9.9")
    assert md.startswith("# Branches")
    assert "Second wording" in md and "Answer two" in md
    assert "First wording" not in md and "Answer one" not in md
    assert "## System\n\nBe brief." in md


def test_markdown_round_trip_keeps_text_and_roles() -> None:
    chat = branching_chat()
    chat["messages"][-1]["content"] = "## User\nlooks like a heading\n\n```\n## Assistant\n```"
    doc = chat_tree.import_markdown(chat_tree.export_markdown(chat, "9.9"))
    assert doc["title"] == "Branches" and doc["system"] == "Be brief."
    assert [m["role"] for m in doc["messages"]] == ["user", "assistant", "user", "assistant"]
    assert doc["messages"][-1]["content"] == chat["messages"][-1]["content"]


def test_markdown_import_needs_sections() -> None:
    with pytest.raises(chat_tree.ChatImportError):
        chat_tree.import_markdown("# Just a title\n\nand some prose")


def test_json_round_trip_keeps_every_branch() -> None:
    exported = chat_tree.export_json(branching_chat(), "9.9")
    assert exported["format"] == chat_tree.FORMAT
    doc = chat_tree.import_json(json.dumps(exported))
    assert len(doc["messages"]) == 6 and doc["leaf"] == "a2b" and doc["system"] == "Be brief."


def test_json_import_accepts_plain_formats() -> None:
    plain = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]
    for text in (json.dumps(plain), json.dumps({"title": "T", "messages": plain})):
        doc = chat_tree.import_json(text)
        assert doc["system"] == "S"
        assert [m["content"] for m in doc["messages"]] == ["Q", "A"]


@pytest.mark.parametrize(
    "text", ["not json", "42", '{"messages": []}', '{"format": "fle-studio-chat", "version": 99, "chat": {}}']
)
def test_json_import_refuses_nonsense(text: str) -> None:
    with pytest.raises(chat_tree.ChatImportError):
        chat_tree.import_json(text)


# ------------------------------------------------------------------------------------------------ store


def test_store_saves_the_tree_and_lists_the_shown_branch(tmp_path: Path) -> None:
    store = chats.Chats(tmp_path)
    saved = store.save(None, {k: v for k, v in branching_chat().items() if k != "id"})
    loaded = store.get(saved["id"])
    assert len(loaded["messages"]) == 6 and loaded["leaf"] == "a2b"
    assert store.list()[0]["messages"] == 4  # the shown branch, not all nodes


def test_store_reads_a_version_one_file(tmp_path: Path) -> None:
    store = chats.Chats(tmp_path)
    (store.dir / "00000000000000aa.json").write_text(
        json.dumps(
            {
                "id": "00000000000000aa",
                "title": "Old",
                "updated": 1,
                "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            }
        ),
        encoding="utf-8",
    )
    chat = store.get("00000000000000aa")
    assert [m["parent"] for m in chat["messages"]] == [None, chat["messages"][0]["id"]]
    assert chat["leaf"] == chat["messages"][1]["id"]


def test_store_search_finds_titles_and_other_branches(tmp_path: Path) -> None:
    store = chats.Chats(tmp_path)
    store.save(None, {k: v for k, v in branching_chat().items() if k != "id"})
    assert store.search("first wording")[0]["in_title"] is False
    assert store.search("BRANCHES")[0]["in_title"] is True
    assert store.search("absent") == [] and store.search("  ") == []


def test_store_pictures_are_files_named_by_hash(tmp_path: Path) -> None:
    store = chats.Chats(tmp_path)
    name = store.put_image(b"\x89PNG\r\n\x1a\nxx", "png")
    assert store.read_image(name) == (b"\x89PNG\r\n\x1a\nxx", "image/png")
    assert store.image_data_url(name).startswith("data:image/png;base64,")  # type: ignore[union-attr]
    with pytest.raises(KeyError):
        store.read_image("../../etc/passwd")
    with pytest.raises(ValueError):
        store.put_image(b"x", "exe")


def test_store_import_turns_inline_pictures_back_into_files(tmp_path: Path) -> None:
    store = chats.Chats(tmp_path)
    name = store.put_image(b"\x89PNG\r\n\x1a\npixels", "png")
    chat = store.save(
        None,
        {
            "title": "Pic",
            "messages": [
                {"role": "user", "content": "look", "attachments": [{"kind": "image", "name": "a.png", "blob": name}]}
            ],
        },
    )
    exported = chat_tree.export_json(chat, "9.9", images=store.image_data_url)
    assert "blob" not in exported["chat"]["messages"][0]["attachments"][0]
    fresh = chats.Chats(tmp_path / "other")
    imported = fresh.import_chat(chat_tree.import_json(json.dumps(exported)))
    assert imported["id"] != chat["id"]
    assert imported["messages"][0]["attachments"][0]["blob"] == name
    assert fresh.read_image(name)[0].endswith(b"pixels")
