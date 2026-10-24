"""The chat endpoints against a real Studio server and the fake llama-server."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chat_support import Stack, make_stack, mcp_config
from test_attachments import _tiny_pdf


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    yield from make_stack(tmp_path)


@pytest.fixture
def vision_stack(tmp_path: Path) -> Iterator[Stack]:
    yield from make_stack(tmp_path, fake_env={"FAKE_VISION": "1"})


@pytest.fixture
def bare_stack(tmp_path: Path) -> Iterator[Stack]:
    """Studio without any engine."""
    yield from make_stack(tmp_path, with_engine=False)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
    },
}


def upload(stack: Stack, name: str, data: bytes) -> tuple[int, Any]:
    return stack.json(
        "POST", f"/api/attachments?name={name}", raw=data, headers={"Content-Type": "application/octet-stream"}
    )


# ------------------------------------------------------------------------------------------------ capabilities


def test_capabilities_without_vision(stack: Stack) -> None:
    status, caps = stack.json("GET", "/api/chat/capabilities")
    assert status == 200
    assert caps["vision"] is False and caps["max_upload_mb"] == 25 and caps["n_ctx"]


def test_capabilities_with_vision(vision_stack: Stack) -> None:
    assert vision_stack.json("GET", "/api/chat/capabilities")[1]["vision"] is True


def test_capabilities_without_an_engine(bare_stack: Stack) -> None:
    status, caps = bare_stack.json("GET", "/api/chat/capabilities")
    assert status == 200 and caps["vision"] is False and caps["tools"] is None


# ------------------------------------------------------------------------------------------------ attachments


def test_upload_text_file(stack: Stack) -> None:
    status, result = upload(stack, "notes.md", "# Titel\nGrüße".encode())
    assert status == 200
    assert (
        result["kind"] == "text" and result["name"] == "notes.md" and "Grüße" in result["text"] and result["tokens"] > 0
    )


def test_upload_name_is_reduced_to_the_file_name(stack: Stack) -> None:
    assert upload(stack, "..%2F..%2Fetc%2Fpasswd", b"x")[1]["name"] == "passwd"


def test_upload_refuses_binary_with_a_reason(stack: Stack) -> None:
    status, result = upload(stack, "tool.exe", b"\x7fELF\x02\x01")
    assert status == 422 and "program" in result["error"]["message"]


def test_upload_limit_is_a_setting(stack: Stack) -> None:
    stack.settings({"chat": {"max_upload_mb": 1}})
    status, result = upload(stack, "big.txt", b"a" * (1024 * 1024 + 10))
    assert status == 413 and "1 MB" in result["error"]["message"] and "Largest upload" in result["error"]["message"]
    assert upload(stack, "ok.txt", b"a" * 1000)[0] == 200
    stack.settings({"chat": {"max_upload_mb": 3}})
    assert upload(stack, "big.txt", b"a" * (2 * 1024 * 1024))[0] == 200  # bigger than the old JSON limit, allowed now


def test_upload_pdf(stack: Stack) -> None:
    pytest.importorskip("pypdf")
    status, result = upload(stack, "paper.pdf", _tiny_pdf("Quantisation notes"))
    assert status == 200 and result["source"] == "pdf" and "Quantisation notes" in result["text"]


def test_picture_without_vision_gets_a_clear_answer(stack: Stack) -> None:
    status, result = upload(stack, "cat.png", PNG)
    assert status == 409 and "cannot look at pictures" in result["error"]["message"]


def test_picture_without_a_model_says_to_load_one(bare_stack: Stack) -> None:
    status, result = upload(bare_stack, "cat.png", PNG)
    assert status == 409 and "Load a model" in result["error"]["message"]


def test_picture_with_vision_is_stored_and_served(vision_stack: Stack) -> None:
    status, result = upload(vision_stack, "cat.png", PNG)
    assert status == 200 and result["kind"] == "image" and result["blob"].endswith(".png")
    code, headers, body = vision_stack.call("GET", f"/api/attachments/{result['blob']}")
    assert code == 200 and headers["content-type"] == "image/png" and body == PNG
    assert vision_stack.call("GET", "/api/attachments/" + "0" * 64 + ".png")[0] == 404
    assert vision_stack.call("GET", "/api/attachments/../../etc/passwd")[0] in (400, 404)


def test_upload_needs_the_session_token(stack: Stack) -> None:
    assert stack.call("POST", "/api/attachments?name=a.txt", raw=b"x", token=False)[0] == 403


# ------------------------------------------------------------------------------------- chats: export, import


def make_chat(stack: Stack) -> dict[str, Any]:
    body = {
        "title": "Gardening plans",
        "model": "m",
        "system": "Be brief.",
        "leaf": "a2b",
        "messages": [
            {"id": "u1", "parent": None, "role": "user", "content": "Which tomatoes?"},
            {"id": "a1", "parent": "u1", "role": "assistant", "content": "San Marzano"},
            {"id": "u2", "parent": "a1", "role": "user", "content": "And cucumbers? (first try)"},
            {"id": "a2", "parent": "u2", "role": "assistant", "content": "Marketmore"},
            {
                "id": "u2b",
                "parent": "a1",
                "role": "user",
                "content": "And peppers?",
                "attachments": [{"kind": "text", "name": "beds.txt", "text": "bed three has shade"}],
            },
            {"id": "a2b", "parent": "u2b", "role": "assistant", "content": "Hungarian wax"},
        ],
    }
    status, chat = stack.json("POST", "/api/chats", body)
    assert status == 200
    return chat  # type: ignore[no-any-return]


def test_chat_keeps_its_branches_through_the_api(stack: Stack) -> None:
    chat = make_chat(stack)
    again = stack.json("GET", f"/api/chats/{chat['id']}")[1]
    assert len(again["messages"]) == 6 and again["leaf"] == "a2b"
    assert stack.json("GET", "/api/chats")[1]["chats"][0]["messages"] == 4


def test_search_finds_text_in_other_branches_and_attachments(stack: Stack) -> None:
    chat = make_chat(stack)
    for needle in ("first try", "shade", "beds.txt", "GARDENING"):
        results = stack.json("GET", f"/api/chats/search?q={needle.replace(' ', '%20')}")[1]["results"]
        assert [r["id"] for r in results] == [chat["id"]], needle
    assert stack.json("GET", "/api/chats/search?q=zebra")[1]["results"] == []
    assert stack.json("GET", "/api/chats/search?q=")[1]["results"] == []


def test_export_json_and_import_it_back(stack: Stack) -> None:
    chat = make_chat(stack)
    status, headers, body = stack.call("GET", f"/api/chats/{chat['id']}/export?format=json")
    assert (
        status == 200 and "attachment" in headers["content-disposition"] and ".json" in headers["content-disposition"]
    )
    doc = json.loads(body)
    assert doc["format"] == "fle-studio-chat" and len(doc["chat"]["messages"]) == 6
    status, imported = stack.json("POST", "/api/chats/import", raw=body)
    assert status == 200 and imported["id"] != chat["id"] and imported["messages"] == 6
    copy = stack.json("GET", f"/api/chats/{imported['id']}")[1]
    assert copy["leaf"] == "a2b" and copy["system"] == "Be brief." and len(copy["messages"]) == 6


def test_export_markdown_and_import_it_back(stack: Stack) -> None:
    chat = make_chat(stack)
    status, headers, body = stack.call("GET", f"/api/chats/{chat['id']}/export?format=md")
    text = body.decode("utf-8")
    assert status == 200 and ".md" in headers["content-disposition"]
    assert "And peppers?" in text and "first try" not in text and "beds.txt" in text
    status, imported = stack.json("POST", "/api/chats/import?format=md", raw=body)
    assert status == 200 and imported["title"] == "Gardening plans" and imported["messages"] == 4


def test_export_keeps_pictures_inline_and_import_makes_files_again(vision_stack: Stack) -> None:
    blob = upload(vision_stack, "cat.png", PNG)[1]["blob"]
    chat = vision_stack.json(
        "POST",
        "/api/chats",
        {
            "title": "Pic",
            "messages": [
                {"role": "user", "content": "look", "attachments": [{"kind": "image", "name": "cat.png", "blob": blob}]}
            ],
        },
    )[1]
    body = vision_stack.call("GET", f"/api/chats/{chat['id']}/export?format=json")[2]
    att = json.loads(body)["chat"]["messages"][0]["attachments"][0]
    assert base64.b64decode(att["data"].split(",", 1)[1]) == PNG and "blob" not in att
    imported = vision_stack.json("POST", "/api/chats/import", raw=body)[1]
    copy = vision_stack.json("GET", f"/api/chats/{imported['id']}")[1]
    assert copy["messages"][0]["attachments"][0]["blob"] == blob


def test_export_unknown_format_and_unknown_chat(stack: Stack) -> None:
    chat = make_chat(stack)
    assert stack.call("GET", f"/api/chats/{chat['id']}/export?format=pdf")[0] == 400
    assert stack.call("GET", "/api/chats/" + "0" * 16 + "/export")[0] == 404


@pytest.mark.parametrize(
    "payload",
    [b"not a chat", b"[]", b'{"format": "fle-studio-chat", "version": 99, "chat": {}}', b"# Title\n\nno sections"],
)
def test_import_refuses_nonsense_with_a_message(stack: Stack, payload: bytes) -> None:
    status, result = stack.json("POST", "/api/chats/import", raw=payload)
    assert status == 400 and result["error"]["message"]


def test_import_accepts_a_plain_openai_style_list(stack: Stack) -> None:
    plain = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    status, imported = stack.json("POST", "/api/chats/import", raw=json.dumps(plain).encode())
    assert status == 200 and imported["messages"] == 2


# ------------------------------------------------------------------------------------------------ tools


def test_tools_listing_defaults(stack: Stack) -> None:
    status, data = stack.json("GET", "/api/tools")
    assert status == 200 and data["enabled"] is True
    assert {t["name"] for t in data["tools"]} == {"calculator", "get_current_time"}


def test_tool_call_endpoint(stack: Stack) -> None:
    status, data = stack.json("POST", "/api/tools/call", {"name": "calculator", "arguments": {"expression": "6*7"}})
    assert status == 200 and data["content"] == "42" and data["is_error"] is False
    assert stack.json("POST", "/api/tools/call", {"name": "rm", "arguments": {}})[0] == 404
    stack.settings({"tools": {"calculator": False}})
    assert stack.json("POST", "/api/tools/call", {"name": "calculator", "arguments": {"expression": "1"}})[0] == 404


def test_confirmation_is_enforced_by_the_server(stack: Stack) -> None:
    """A page that forgets to ask still cannot run a tool that needs approval."""

    def reload_and_wait() -> None:
        stack.json("POST", "/api/tools/reload")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            data = stack.json("GET", "/api/tools")[1]
            if not data["connecting"] and any(t["name"] == "fake__write_note" for t in data["tools"]):
                return
            time.sleep(0.1)
        pytest.fail(f"the MCP server never became ready: {data['servers']}")

    stack.settings({"tools": {"mcp_enabled": True, "mcp_json": mcp_config()}})
    reload_and_wait()
    call = {"name": "fake__write_note", "arguments": {"text": "hello"}}
    status, result = stack.json("POST", "/api/tools/call", call)
    assert status == 409 and "approval" in result["error"]["message"]
    assert stack.json("POST", "/api/tools/call", {**call, "approved": "yes"})[0] == 409  # only a real true counts
    status, result = stack.json("POST", "/api/tools/call", {**call, "approved": True})
    assert status == 200 and result["content"] == "saved note 1"
    # the JSON may say "autoApprove"; that is ignored, only the setting in Studio trusts a server
    stack.settings(
        {
            "tools": {
                "mcp_json": json.dumps(
                    {
                        "mcpServers": {
                            "fake": {**json.loads(mcp_config())["mcpServers"]["fake"], "autoApprove": ["write_note"]}
                        }
                    }
                )
            }
        }
    )
    reload_and_wait()
    assert stack.json("POST", "/api/tools/call", call)[0] == 409  # still asks
    stack.settings({"tools": {"mcp_trusted": ["fake"]}})
    assert stack.json("POST", "/api/tools/call", call)[1]["content"] == "saved note 1"  # trusted: no question


# ------------------------------------------------------------------------------------- the proxy and the trap


def test_tools_with_a_forced_format_are_refused_by_the_proxy(stack: Stack) -> None:
    for fmt in (
        {"response_format": {"type": "json_object"}},
        {"json_schema": {"type": "object"}},
        {"grammar": "root ::= x"},
    ):
        status, data = stack.chat({"messages": [{"role": "user", "content": "hi"}], "tools": [TOOL], **fmt})
        assert status == 400, fmt
        assert "tools" in data["error"]["message"] and next(iter(fmt)) in data["error"]["message"]


def test_the_proxy_still_serves_tools_alone_and_formats_alone(stack: Stack) -> None:
    status, data = stack.chat(
        {"messages": [{"role": "user", "content": "plain"}], "response_format": {"type": "json_object"}}
    )
    assert status == 200 and json.loads(data["choices"][0]["message"]["content"]) is not None
    status, data = stack.chat({"messages": [{"role": "user", "content": "plain"}], "tools": [TOOL]})
    assert status == 200 and data["choices"][0]["message"]["content"]


def test_tool_call_round_trip_through_the_fake_engine(stack: Stack) -> None:
    ask = [{"role": "user", "content": 'CALL_TOOL calculator {"expression": "6*7"}'}]
    status, data = stack.chat({"messages": ask, "tools": [TOOL]})
    assert status == 200 and data["choices"][0]["finish_reason"] == "tool_calls"
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "calculator" and json.loads(call["function"]["arguments"]) == {
        "expression": "6*7"
    }
    result = stack.json("POST", "/api/tools/call", {"name": "calculator", "arguments": call["function"]["arguments"]})[
        1
    ]
    follow = [
        *ask,
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": result["content"]},
    ]
    status, data = stack.chat({"messages": follow, "tools": [TOOL]})
    assert (
        status == 200
        and "Tool said" in data["choices"][0]["message"]["content"]
        and "42" in data["choices"][0]["message"]["content"]
    )


def test_streamed_tool_calls_arrive_as_deltas(stack: Stack) -> None:
    status, events = stack.chat(
        {"messages": [{"role": "user", "content": 'CALL_TOOL calculator {"expression": "1+1"}'}], "tools": [TOOL]},
        stream=True,
    )
    assert status == 200
    deltas = [c for e in events for ch in e.get("choices", []) for c in ch.get("delta", {}).get("tool_calls", [])]
    assert deltas and deltas[0]["function"]["name"] == "calculator"
    arguments = "".join(d["function"].get("arguments", "") for d in deltas)
    assert json.loads(arguments) == {"expression": "1+1"}


def test_the_fake_engine_does_not_take_pictures_without_vision(stack: Stack) -> None:
    part = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
    ]
    status, data = stack.chat({"messages": [{"role": "user", "content": part}]})
    assert status == 400 and "image input is not supported" in json.dumps(data)


def test_the_fake_engine_takes_pictures_with_vision(vision_stack: Stack) -> None:
    part = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
    ]
    status, data = vision_stack.chat({"messages": [{"role": "user", "content": part}]})
    assert status == 200 and data["choices"][0]["message"]["content"]
