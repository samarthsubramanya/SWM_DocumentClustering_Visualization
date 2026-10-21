"""Corpus loaders. Each loader returns a list[dict] with at least {id, text}."""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
from pathlib import Path

from swm import db

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Same archive scikit-learn's fetch_20newsgroups downloads (20news-bydate, 18846 docs).
ARCHIVE_URL = "https://ndownloader.figshare.com/files/5975967"
ARCHIVE_SHA256 = "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610"
DEFAULT_ARCHIVE = RAW / "20news-bydate.tar.gz"

# Posts that are mostly uuencoded/binhex attachments (~22 in 20NG) aren't text: they hang
# LLM extraction and pollute cluster keywords. Share of characters in >25-char tokens.
MAX_ENCODED_SHARE = 0.65


# ==========================================================
# Archive
# ==========================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_archive(path: Path = DEFAULT_ARCHIVE) -> Path:
    """Use the local 20news-bydate archive; download it only when missing. Checksum-verified."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"Downloading 20news-bydate to {path} ...")
        urllib.request.urlretrieve(ARCHIVE_URL, tmp)
        tmp.rename(path)
    if (digest := _sha256(path)) != ARCHIVE_SHA256:
        raise ValueError(f"{path} checksum {digest} != expected {ARCHIVE_SHA256}; delete it to re-download.")
    return path


# ==========================================================
# Cleaning (same rules as scikit-learn's remove=('headers', 'footers', 'quotes'))
# ==========================================================

_QUOTE_RE = re.compile(r"(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)")


def strip_header(text: str) -> str:
    return text.partition("\n\n")[2]


def strip_quotes(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _QUOTE_RE.search(line))


def strip_footer(text: str) -> str:
    """Drop everything after the last line made only of dashes/blank (signature block)."""
    lines = text.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().strip("-") == "":
            return "\n".join(lines[:i]) if i > 0 else text
    return text


def encoded_share(text: str) -> float:
    tokens = text.split()
    return sum(len(t) for t in tokens if len(t) > 25) / max(sum(len(t) for t in tokens), 1)


# ==========================================================
# Loader
# ==========================================================

def load_20ng(subset: str = "all", min_tokens: int = 20, archive: Path = DEFAULT_ARCHIVE) -> list[dict]:
    """Load 20news-bydate straight from the tar.gz, strip headers/footers/quotes,
    drop short posts and encoded attachments.

    subset in {'train', 'test', 'all'}. Labels are indices into the sorted newsgroup names,
    matching scikit-learn's target numbering.
    """
    path = ensure_archive(archive)
    raw = []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")  # 20news-bydate-train/sci.space/60151
            if not member.isfile() or len(parts) != 3:
                continue
            split = parts[0].rsplit("-", 1)[-1]
            if subset != "all" and split != subset:
                continue
            text = tar.extractfile(member).read().decode("latin-1")
            raw.append((split, parts[1], parts[2], text))

    groups = sorted({g for _, g, _, _ in raw})
    docs = []
    for split, group, number, text in raw:
        text = strip_footer(strip_quotes(strip_header(text))).replace("\x00", "").strip()  # Postgres rejects NUL
        if len(text.split()) < min_tokens or encoded_share(text) > MAX_ENCODED_SHARE:
            continue
        docs.append({
            "id": f"{split}/{group}/{number}",
            "text": text,
            "label": groups.index(group),
            "label_name": group,
            "split": split,
            "source": "20ng",
        })
    return docs


def save_documents(docs: list[dict]) -> None:
    """Upsert docs and delete rows of the same source that the loader no longer yields."""
    for source in {d.get("source", "20ng") for d in docs}:
        db.execute(
            "DELETE FROM documents WHERE source = %s AND NOT (id = ANY(%s))",
            (source, [d["id"] for d in docs if d.get("source", "20ng") == source]),
        )
    db.executemany(
        """INSERT INTO documents (id, text, label, label_name, split, source)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, label = EXCLUDED.label,
               label_name = EXCLUDED.label_name, split = EXCLUDED.split, source = EXCLUDED.source""",
        [(d["id"], d["text"], d.get("label"), d.get("label_name"), d.get("split"), d.get("source", "20ng"))
         for d in docs],
    )


def load_documents(source: str = "20ng") -> list[dict]:
    """Documents ordered by id — every downstream array is aligned to this order."""
    return db.query(
        "SELECT id, text, label, label_name FROM documents WHERE source = %s ORDER BY id",
        (source,),
    )
