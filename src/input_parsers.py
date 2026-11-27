import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import Any


class _HTMLTextParser(HTMLParser):
    _blocks = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    _ignored = {"script", "style", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._ignored:
            self.ignored_depth += 1
        if self.ignored_depth:
            return
        if tag in self._blocks:
            self.parts.append("\n")
        for name, value in attrs:
            if value and ((tag == "a" and name == "href") or (tag == "form" and name == "action")):
                self.parts.append(f" [link: {value}] ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and tag in self._blocks:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def parse_html_text(raw_bytes: bytes) -> str:
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        html = raw_bytes.decode("utf-16")
    else:
        charset = re.search(rb'<meta\b[^>]*charset\s*=\s*["\']?\s*([\w.-]+)', raw_bytes[:4096], re.I)
        if charset:
            html = raw_bytes.decode(charset.group(1).decode("ascii"))
        else:
            try:
                html = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                html = raw_bytes.decode("windows-1252")
    parser = _HTMLTextParser()
    parser.feed(html)
    parser.close()
    text = re.sub(r"[ \t\f\v]+", " ", "".join(parser.parts))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_json_samples(raw_bytes: bytes) -> list[dict[str, Any]]:
    data = json.loads(raw_bytes.decode("utf-8-sig"))

    if isinstance(data, dict):
        data = data.get("samples", data.get("data", [data]))

    if not isinstance(data, list):
        data = [data]

    samples = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            text = item
            metadata = {}
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("email")
                or item.get("payload")
                or item.get("raw_payload")
                or ""
            )
            metadata = item
        else:
            raise ValueError(
                f"JSON sample {index} must be a string or an object, "
                f"not {type(item).__name__}."
            )

        if str(text).strip():
            samples.append({
                "id": f"sample-{index}",
                "text": str(text).strip(),
                "metadata": metadata,
            })

    return samples


def parse_csv_samples(raw_bytes: bytes) -> list[dict[str, Any]]:
    content = raw_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))

    samples = []
    for index, row in enumerate(reader, start=1):
        text = (
            row.get("text")
            or row.get("email")
            or row.get("payload")
            or row.get("raw_payload")
            or next(iter(row.values()), "")
        )

        if str(text).strip():
            samples.append({
                "id": f"sample-{index}",
                "text": str(text).strip(),
                "metadata": dict(row),
            })

    return samples
