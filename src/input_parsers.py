import csv
import io
import json
from typing import Any


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
