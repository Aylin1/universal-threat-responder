"""Metadata collected from the actual fitted models and evaluation configuration."""
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Cannot serialize model parameter of type {type(value).__name__}")


def _estimator_info(estimator):
    return {
        "class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "parameters": _json_value(estimator.get_params(deep=False)),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    names = ("scikit-learn", "xgboost", "numpy", "pandas", "langchain-ollama",
             "langgraph", "langchain-milvus", "langchain-huggingface", "pymilvus")
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def build_run_metadata(*, run_id, mode, args, config, loader, adapter, history_manifest, sample_count):
    classifier = loader.ml_classifier
    calibration_params = classifier.get_params(deep=False).copy()
    base = calibration_params.pop("estimator")
    ioc_path = Path(adapter.ioc_index_path) if adapter.ioc_index_path else None
    ioc = {"path": str(ioc_path) if ioc_path else None, "sha256": None, "fetched_at": None}
    if ioc_path and ioc_path.exists():
        ioc["sha256"] = _sha256(ioc_path)
        ioc["fetched_at"] = json.loads(ioc_path.read_text(encoding="utf-8")).get("fetched_at")
    return {
        "run_id": run_id,
        "metadata_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "cli_arguments": vars(args),
        "config": asdict(config),
        "dataset": {"path": loader.file_path, "sha256": _sha256(loader.file_path),
                    "seed": loader.seed, "outer_fold": loader.outer_fold,
                    "sample_count": sample_count, "split_fingerprint": loader.split_fingerprint()},
        "classifier": {"calibrator": {"class": f"{type(classifier).__module__}.{type(classifier).__qualname__}",
                                      "parameters": _json_value(calibration_params)},
                       "base_estimator": _estimator_info(base),
                       "vectorizer": _estimator_info(loader.vectorizer)},
        "conformal": loader.router.to_dict(),
        "llm": {"provider": config.llm.provider, "model": config.llm.model_name,
                "temperature": config.llm.temperature, "keep_alive": config.llm.keep_alive,
                "draws": args.llm_samples, "confidence_threshold": config.llm.confidence_threshold,
                "model_digest": None},
        "retrieval": {"embedding_model": adapter.embeddings.model_name,
                      "history_collection": adapter.history_collection,
                      "pattern_collection": adapter.pattern_collection,
                      "k_history": adapter.k_history, "k_patterns": adapter.k_patterns,
                      "distance_metric": adapter.distance_metric, "guard": asdict(adapter.guard_config),
                      "history_manifest": history_manifest, "ioc_snapshot": ioc},
        "environment": {"python": platform.python_version(), "packages": _versions()},
    }
