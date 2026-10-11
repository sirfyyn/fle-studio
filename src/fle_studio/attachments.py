"""Files attached to a chat message: text and code, PDF (text layer) and pictures.

Nothing here talks to the network. PDF text needs the optional pure-Python library `pypdf`
(`pip install "fle-studio[pdf]"`); without it, a PDF is refused with a message that says so.
"""

from __future__ import annotations

import io
import math
from typing import Any

MAX_TEXT_CHARS = 3_000_000  # one attachment; the rest is cut off and the message says so
MAX_PDF_PAGES = 2000
IMAGE_TOKENS = 600  # a rough cost of one picture in the context; models differ a lot

_IMAGES = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
)
_BINARY_HINTS = (
    (
        b"PK\x03\x04",
        "Office documents (docx, xlsx, pptx) and archives are not supported; save the content as text or PDF.",
    ),
    (b"\x7fELF", "That is a program, not a document."),
    (b"\x1f\x8b", "Compressed files are not supported; unpack it first."),
    (b"SQLite format 3", "Database files are not supported."),
)


class AttachmentError(ValueError):
    """The file cannot be attached; the message tells the user why."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def estimate_tokens(text: str) -> int:
    """A rough token count without a tokenizer: about 3.6 characters per token for ASCII text, one per other
    character. Real counts depend on the model; this is for the context warning only. (chat.js has the same rule.)"""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return max(1, math.ceil(ascii_chars / 3.6 + (len(text) - ascii_chars) * 0.9)) if text else 0


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """(mime type, file extension) when the bytes are a PNG, JPEG, GIF or WebP picture."""
    for magic, mime, ext in _IMAGES:
        if data.startswith(magic):
            return mime, ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return None


def _decode(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # Old Windows or Latin-1 files: accept them when almost every byte is printable text.
    odd = sum(1 for b in data[:20000] if b < 9 or 13 < b < 32 or b == 127)
    if odd > max(2, len(data[:20000]) // 50):
        raise AttachmentError("That file does not look like text.")
    return data.decode("cp1252", "replace")


def extract_text(name: str, data: bytes) -> dict[str, Any]:
    """Text or code file → {"kind": "text", ...}."""
    for magic, why in _BINARY_HINTS:
        if data.startswith(magic):
            raise AttachmentError(why)
    if b"\x00" in data[:8000] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise AttachmentError("That is a binary file; attach text, code, a PDF or a picture.")
    text = _decode(data).replace("\r\n", "\n").replace("\r", "\n")
    return _text_result(name, text, "text")


def _text_result(name: str, text: str, source: str, pages: int | None = None) -> dict[str, Any]:
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
    out: dict[str, Any] = {
        "kind": "text",
        "name": name,
        "text": text,
        "chars": len(text),
        "tokens": estimate_tokens(text),
        "truncated": truncated,
        "source": source,
    }
    if pages is not None:
        out["pages"] = pages
    return out


def extract_pdf(name: str, data: bytes) -> dict[str, Any]:
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
    except ImportError:
        raise AttachmentError(
            'Reading PDFs needs the small "pypdf" library. Install it with: '
            'pip install "fle-studio[pdf]"   (or: pip install pypdf)',
            501,
        ) from None
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise AttachmentError("This PDF is password protected.")
        total = len(reader.pages)
        parts = []
        for number, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                parts.append(f"[Page {number}]\n{page_text}")
    except AttachmentError:
        raise
    except Exception as exc:  # pypdf raises many kinds of errors for damaged files
        raise AttachmentError(f"Could not read this PDF: {type(exc).__name__}: {exc}") from None
    if not parts:
        raise AttachmentError("This PDF has no text layer (a scan?). FLE Studio does not do text recognition.")
    result = _text_result(name, "\n\n".join(parts), "pdf", total)
    if total > MAX_PDF_PAGES:
        result["truncated"] = True
    return result


def extract(name: str, data: bytes) -> dict[str, Any]:
    """Classify an uploaded file. Pictures come back as {"kind": "image", "mime", "ext"} for the caller to store."""
    if not data:
        raise AttachmentError("That file is empty.")
    image = sniff_image(data)
    if image:
        return {
            "kind": "image",
            "name": name,
            "mime": image[0],
            "ext": image[1],
            "size": len(data),
            "tokens": IMAGE_TOKENS,
        }
    if data.startswith(b"%PDF-") or name.lower().endswith(".pdf"):
        return extract_pdf(name, data)
    return extract_text(name, data)
