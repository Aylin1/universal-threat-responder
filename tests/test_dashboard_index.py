import pytest

from src.dashboard_index import prepare_dashboard_index


def make_index(path):
    for collection in ("email_history", "social_engineering_patterns"):
        manifest = path / "collections" / collection / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(collection, encoding="utf-8")
    (path / "history_manifest.json").write_text("provenance", encoding="utf-8")


def test_index_is_copied_once_and_reused(tmp_path):
    source, target = tmp_path / "source", tmp_path / "data" / "dashboard_rag.db"
    make_index(source)

    assert prepare_dashboard_index(source, target) == target
    copied = target / "collections" / "email_history" / "manifest.json"
    assert copied.read_text(encoding="utf-8") == "email_history"
    copied.write_text("existing", encoding="utf-8")
    assert prepare_dashboard_index(source, target) == target
    assert copied.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("existing", [False, True])
def test_missing_index_is_reported_without_rebuilding(tmp_path, existing):
    source, target = tmp_path / "source", tmp_path / "dashboard_rag.db"
    if existing:
        make_index(source)
        target.mkdir()
    with pytest.raises(RuntimeError, match="retrieval index.*incomplete"):
        prepare_dashboard_index(source, target)
    if not existing:
        assert not target.exists()


def test_changed_prebuilt_index_does_not_silently_reuse_old_split(tmp_path):
    source, target = tmp_path / "source", tmp_path / "dashboard_rag.db"
    make_index(source)
    prepare_dashboard_index(source, target)
    (source / "history_manifest.json").write_text("different split", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the prebuilt index"):
        prepare_dashboard_index(source, target)


def test_failed_copy_does_not_leave_partial_index(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "data" / "dashboard_rag.db"
    make_index(source)

    def fail_copy(*args, **kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr("src.dashboard_index.shutil.copytree", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        prepare_dashboard_index(source, target)
    assert not target.exists()
