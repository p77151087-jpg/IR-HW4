"""Parse the ORIGINAL SGML: ModApte is defined by attributes, not nonempty labels."""
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import tarfile
import urllib.request

from .config import DATA, DATA_URL, README_URL
from .io import read_json, sha256, write_json


class ReutersParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.documents = []
        self.current = None
        self.field = None
        self.buffer = []
        self.in_text = False
        self.raw_text = []
        self.topic = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "reuters":
            self.current = {"id": attrs["newid"], "topics_flag": attrs.get("topics", ""),
                            "lewis_split": attrs.get("lewissplit", ""), "title": "", "body": "",
                            "date": "", "topics": [], "text_type": "NORM"}
        elif self.current is not None:
            if tag == "text":
                self.in_text = True
                self.raw_text = []
                self.current["text_type"] = attrs.get("type", "NORM")
            if tag in {"title", "body", "date", "topics"}:
                self.field, self.buffer = tag, []
            elif tag == "d" and self.field == "topics":
                self.topic = []

    def handle_data(self, data):
        if self.current is not None:
            if self.in_text:
                self.raw_text.append(data)
            if self.field:
                self.buffer.append(data)
            if self.topic is not None:
                self.topic.append(data)

    def handle_endtag(self, tag):
        if self.current is None:
            return
        if tag == "d" and self.topic is not None:
            self.current["topics"].append("".join(self.topic).strip())
            self.topic = None
        if tag == self.field:
            if tag != "topics":
                self.current[tag] = "".join(self.buffer).strip()
            self.field, self.buffer = None, []
        if tag == "text":
            self.in_text = False
            # UNPROC records may contain no TITLE/BODY tags; retain their TEXT content.
            if not self.current["body"] and not self.current["title"]:
                self.current["body"] = "".join(self.raw_text).strip()
        if tag == "reuters":
            doc = self.current
            doc["topics"] = sorted(set(filter(None, doc["topics"])))
            doc["split"] = ("train" if doc["lewis_split"] == "TRAIN" else "test") if (
                doc["topics_flag"] == "YES" and doc["lewis_split"] in {"TRAIN", "TEST"}) else "unused"
            self.documents.append(doc)
            self.current = None


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "IR-HW4 academic course project"})
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(path)


def prepare_data():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        return load_data()[1]
    archive, readme = DATA / "reuters21578.tar.gz", DATA / "README.original.txt"
    if not archive.exists():
        download(DATA_URL, archive)
    if not readme.exists():
        download(README_URL, readme)
    documents, sources = [], {}
    raw = DATA / "raw"
    raw.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        members = sorted((m for m in package.getmembers() if Path(m.name).suffix == ".sgm"), key=lambda m: m.name)
        if len(members) != 22:
            raise ValueError("Expected 22 original SGML members")
        for member in members:
            # No extractall: write only the checked SGML basenames inside this project.
            name = Path(member.name).name
            if not name.startswith("reut2-"):
                raise ValueError("Unexpected SGML filename")
            content = package.extractfile(member).read()
            target = raw / name
            target.write_bytes(content)
            sources["raw/" + name] = sha256(target)
            parser = ReutersParser()
            parser.feed(content.decode("latin-1"))
            parser.close()
            documents.extend(parser.documents)
    documents.sort(key=lambda d: int(d["id"]))
    counts = Counter(d["split"] for d in documents)
    if len({d["id"] for d in documents}) != 21578 or counts != {"train": 9603, "test": 3299, "unused": 8676}:
        raise ValueError(f"Unexpected corpus or ModApte counts: {counts}")
    target = DATA / "documents.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for doc in documents:
            stream.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "dataset": "Reuters-21578 Distribution 1.0", "split": "ModApte",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": DATA_URL, "readme_url": README_URL,
        "archive_sha256": sha256(archive), "readme_sha256": sha256(readme),
        "documents_sha256": sha256(target), "raw_sha256": sources,
        "counts": dict(counts), "documents": len(documents),
        "rule": "TOPICS=YES and LEWISSPLIT=TRAIN/TEST; do not filter empty TOPICS elements.",
        "text_policy": "TITLE plus BODY; TITLE only for brief records; unstructured TEXT fallback; keep empty text as zero vectors.",
        "label_policy": "Original TOPICS elements, multilabel; never use TOPICS flag as a category.",
        "license_note": "Reuters original README permits distribution for research purposes only; local coursework use.",
    }
    write_json(manifest_path, manifest)
    return manifest


def load_data():
    manifest = read_json(DATA / "manifest.json")
    path = DATA / "documents.jsonl"
    if sha256(path) != manifest["documents_sha256"]:
        raise ValueError("Reuters corpus SHA256 mismatch; refuse modified data")
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(docs) != 21578 or len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Corpus count or duplicate IDs")
    counts = Counter(d["split"] for d in docs)
    if counts != {"train": 9603, "test": 3299, "unused": 8676}:
        raise ValueError("ModApte split mismatch")
    return docs, manifest
