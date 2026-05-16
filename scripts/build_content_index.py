from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"
OUT_PATH = CONTENT_DIR / "index.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_frontmatter(raw: str) -> dict:
    out: dict = {}
    list_key: str | None = None
    for line in raw.splitlines():
        if list_key is not None and line.startswith("  - "):
            out[list_key].append(line[4:].strip())
            continue
        list_key = None
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v == "":
            out[k] = []
            list_key = k
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            out[k] = [x.strip() for x in inner.split(",")] if inner else []
        elif v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            out[k] = v
    return out


def parse_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path}: missing frontmatter (expected --- block at top)")
    fm_raw, body = m.groups()
    entry = parse_frontmatter(fm_raw)
    entry["body"] = body.strip()
    for required in ("id", "level", "title", "short"):
        if required not in entry:
            raise ValueError(f"{path}: missing required field '{required}'")
    return entry


def collect(subdir: str) -> dict:
    out: dict = {}
    d = CONTENT_DIR / subdir
    if not d.exists():
        return out
    for path in sorted(d.glob("*.md")):
        entry = parse_md(path)
        out[entry["id"]] = entry
    return out


def main() -> None:
    components = collect("components")
    concepts = collect("concepts")
    payload = {"components": components, "concepts": concepts}
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {OUT_PATH.relative_to(ROOT)} · {len(components)} components · {len(concepts)} concepts")


if __name__ == "__main__":
    main()
