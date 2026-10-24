from __future__ import annotations

import sys

import pytest

from fle_studio import attachments


def test_text_files_in_common_encodings() -> None:
    assert attachments.extract("a.py", b"print('hi')\r\n")["text"] == "print('hi')\n"
    assert attachments.extract("b.txt", "﻿grüße".encode())["text"] == "grüße"
    assert attachments.extract("c.txt", "hallo".encode("utf-16"))["text"] == "hallo"
    assert attachments.extract("d.txt", "Größe".encode("cp1252"))["text"] == "Größe"


def test_text_result_has_a_token_estimate() -> None:
    result = attachments.extract("a.txt", b"x" * 3600)
    assert result["kind"] == "text" and result["chars"] == 3600 and result["tokens"] == 1000
    assert result["truncated"] is False


def test_huge_text_is_cut_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attachments, "MAX_TEXT_CHARS", 100)
    result = attachments.extract("a.txt", b"y" * 500)
    assert result["truncated"] is True and result["chars"] == 100


@pytest.mark.parametrize(
    ("data", "word"),
    [
        (b"", "empty"),
        (b"PK\x03\x04rest", "Office"),
        (b"\x7fELF\x02", "program"),
        (b"\x1f\x8b\x08", "Compressed"),
        (b"abc\x00\x01\x02def", "binary"),
    ],
)
def test_refused_files_explain_why(data: bytes, word: str) -> None:
    with pytest.raises(attachments.AttachmentError) as err:
        attachments.extract("f.bin", data)
    assert word in str(err.value)


def test_pictures_are_recognised_by_content() -> None:
    png = attachments.extract("x.txt", b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    assert (png["kind"], png["mime"], png["ext"], png["tokens"]) == (
        "image",
        "image/png",
        "png",
        attachments.IMAGE_TOKENS,
    )
    assert attachments.sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == ("image/webp", "webp")
    assert attachments.sniff_image(b"plain text") is None


def test_estimate_tokens() -> None:
    assert attachments.estimate_tokens("") == 0
    assert attachments.estimate_tokens("a") == 1
    assert attachments.estimate_tokens("ü" * 10) == 9


def test_pdf_without_pypdf_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `import pypdf` raise ImportError
    with pytest.raises(attachments.AttachmentError) as err:
        attachments.extract("doc.pdf", b"%PDF-1.4 whatever")
    assert err.value.status == 501 and "pypdf" in str(err.value)


def _tiny_pdf(text: str) -> bytes:
    """A one-page PDF with a text layer, built by hand (no library needed)."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode()
    objs[3] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_pdf_text_layer() -> None:
    pytest.importorskip("pypdf")
    result = attachments.extract("doc.pdf", _tiny_pdf("Hello PDF world"))
    assert result["source"] == "pdf" and result["pages"] == 1
    assert "Hello PDF world" in result["text"] and "[Page 1]" in result["text"]


def test_broken_pdf_is_a_clean_error() -> None:
    pytest.importorskip("pypdf")
    with pytest.raises(attachments.AttachmentError):
        attachments.extract("doc.pdf", b"%PDF-1.4 this is not a pdf")
