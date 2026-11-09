"""HW3's Unicode normalization/token pattern with a local frozen stopword list."""
from dataclasses import dataclass
from functools import lru_cache
import unicodedata
import regex

from .config import ROOT
from .io import read_json

TOKEN_RE = regex.compile(r"\p{N}+(?:\.\p{N}+)+|[\p{L}\p{N}][\p{L}\p{M}\p{N}]*(?:[-‐‑'’][\p{L}\p{N}][\p{L}\p{M}\p{N}]*)*")


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str
    rule: str


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold().replace("’", "'").replace("‐", "-").replace("‑", "-")


@lru_cache(maxsize=1)
def stopwords():
    return frozenset(read_json(ROOT / "resources" / "stopwords.json")["words"])


def terms(text):
    # Repeated query words remain repeated; document and query preprocessing is identical.
    return [word for match in TOKEN_RE.finditer(text) if (word := normalize(match.group())) not in stopwords()]


def document_blocks(document):
    title, body = document.get("title", "").strip(), document.get("body", "").strip()
    return [part for part in (title, body) if part]


def document_text(document):
    return "\n".join(document_blocks(document))
