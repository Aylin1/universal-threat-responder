"""Install the prebuilt dashboard retrieval index without rebuilding it."""

from pathlib import Path
import shutil
import tempfile


def prepare_dashboard_index(source: Path, target: Path) -> Path:
    required = (
        Path("collections/email_history/manifest.json"),
        Path("collections/social_engineering_patterns/manifest.json"),
        Path("history_manifest.json"),
    )
    missing = [str(path) for path in required if not (source / path).is_file()]
    if missing:
        raise RuntimeError(f"Prebuilt retrieval index at {source} is incomplete: {', '.join(missing)}")

    if target.exists():
        missing = [str(path) for path in required if not (target / path).is_file()]
        if missing:
            raise RuntimeError(f"Dashboard retrieval index at {target} is incomplete: {', '.join(missing)}")
        if (source / "history_manifest.json").read_bytes() != (target / "history_manifest.json").read_bytes():
            raise RuntimeError(
                f"Dashboard retrieval index at {target} does not match the prebuilt index. "
                "Stop the dashboard and restore the matching index before retrying."
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dashboard-rag-", dir=target.parent) as temp:
        prepared = Path(temp) / target.name
        shutil.copytree(source, prepared)
        prepared.rename(target)
    return target
