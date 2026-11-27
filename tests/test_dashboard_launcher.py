from contextlib import nullcontext
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import requests

from scripts import launch_dashboard as launcher


def test_missing_classifier_does_not_trigger_training(tmp_path, monkeypatch):
    from config import cfg

    monkeypatch.setattr(cfg.dataset, "classifier_path", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(launcher, "run", lambda command: pytest.fail("Must not train at dashboard startup"))
    with pytest.raises(RuntimeError, match="Saved classifier not found"):
        launcher.prepare_classifier()


def test_prepare_ollama_reuses_installed_model(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NAME ID\nllama3:latest abc\n", stderr=""),
    )
    monkeypatch.setattr(launcher, "run", lambda command: pytest.fail(f"Unexpected download: {command}"))
    launcher.prepare_ollama()


def test_prepare_ollama_downloads_missing_model(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NAME ID\n", stderr=""),
    )
    calls = []
    monkeypatch.setattr(launcher, "run", calls.append)
    launcher.prepare_ollama()
    assert calls == [["ollama", "pull", "llama3"]]


def test_configured_ollama_host_is_not_restarted(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "remote.example:11434")
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "ollama")
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="connection failed"),
    )
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not start a server"),
    )
    with pytest.raises(RuntimeError, match="configured OLLAMA_HOST"):
        launcher.prepare_ollama()


def test_container_ollama_reuses_existing_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"models": [{"name": "llama3:latest"}]})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: pytest.fail("Must not download an existing model"))
    launcher.prepare_container_ollama()


def test_container_ollama_downloads_missing_model_and_reports_failure(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"models": []})
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "success"})

    monkeypatch.setattr(requests, "post", post)
    launcher.prepare_container_ollama()
    assert calls == [("http://ollama:11434/api/pull", {"model": "llama3", "stream": False})]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"error": "download failed"},
    ))
    with pytest.raises(RuntimeError, match="download failed"):
        launcher.prepare_container_ollama()


def test_container_ollama_reports_connection_failure(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")

    def fail(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(RuntimeError, match="Check the Ollama service.*connection refused"):
        launcher.prepare_container_ollama()


def test_dependency_install_is_reused_and_only_marked_after_success(tmp_path, monkeypatch):
    env = tmp_path / "env"
    python = env / "Scripts" / "python.exe"
    marker = env / "dashboard-requirements.sha256"
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "VENV", env)
    monkeypatch.setattr(launcher, "PYTHON", python)
    monkeypatch.setattr(launcher, "MARKER", marker)
    (tmp_path / "requirements.txt").write_text("streamlit\n")

    def create(path, with_pip):
        python.parent.mkdir(parents=True)
        python.touch()

    monkeypatch.setattr(launcher.venv, "create", create)
    calls = []
    monkeypatch.setattr(launcher, "run", calls.append)
    launcher.prepare_environment()
    launcher.prepare_environment()
    assert sum("install" in command for command in calls) == 1
    (tmp_path / "requirements.txt").write_text("streamlit\npytest\n")
    def fail_install(command):
        if "install" in command:
            raise RuntimeError("install failed")

    monkeypatch.setattr(launcher, "run", fail_install)
    old_marker = marker.read_text()
    with pytest.raises(RuntimeError, match="install failed"):
        launcher.prepare_environment()
    assert marker.read_text() == old_marker


def test_select_environment_reuses_existing_environments(tmp_path):
    dashboard = tmp_path / ".dashboard-venv"
    project = tmp_path / ".venv"
    assert launcher.select_environment(tmp_path) == dashboard
    project.mkdir()
    assert launcher.select_environment(tmp_path) == project
    dashboard.mkdir()
    assert launcher.select_environment(tmp_path) == dashboard


def test_launcher_prepares_before_starting_streamlit(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.sys, "argv", ["launch_dashboard"])
    monkeypatch.setattr(launcher, "setup_lock", nullcontext)
    calls = []
    monkeypatch.setattr(launcher, "prepare_environment", lambda: calls.append("environment"))
    monkeypatch.setattr(launcher, "prepare_ollama", lambda: calls.append("ollama"))
    monkeypatch.setattr(launcher, "run", lambda command: calls.append(command[1:]))
    launcher.main()
    assert calls == [
        "environment", "ollama", ["-m", "scripts.launch_dashboard", "--check-classifier"],
        ["-m", "streamlit", "run", "app.py"],
    ]


def test_container_prepares_before_starting_streamlit(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher.sys, "argv", ["launch_dashboard", "--container"])
    calls = []
    monkeypatch.setattr(launcher, "prepare_container_ollama", lambda: calls.append("ollama"))
    monkeypatch.setattr(launcher, "run", lambda command: calls.append(command[1:]))
    launcher.main()
    assert calls == [
        "ollama", ["-m", "scripts.launch_dashboard", "--check-classifier"],
        [
            "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0",
            "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false",
        ],
    ]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_waiting_message_repeats_and_stops_on_completion_or_failure(
    tmp_path, monkeypatch, capsys, exit_code,
):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "STATUS_INTERVAL_SECONDS", 0.03)
    reporters = []

    def create_reporter(**kwargs):
        reporter = threading.Thread(**kwargs)
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(launcher, "threading", SimpleNamespace(Event=threading.Event, Thread=create_reporter))
    expected = pytest.raises(subprocess.CalledProcessError) if exit_code else nullcontext()
    with expected:
        with launcher.waiting_for("Installing packages"):
            launcher.run([sys.executable, "-c", f"import time; time.sleep(0.15); raise SystemExit({exit_code})"])
    output = capsys.readouterr().out
    assert "Still waiting for Installing packages" in output
    assert "Please keep this window open" in output
    assert ("Finished: Installing packages." in output) == (exit_code == 0)
    assert len(reporters) == 1
    assert not reporters[0].is_alive()


@pytest.mark.skipif(os.name != "nt", reason="The launcher uses Windows file locking")
def test_concurrent_setup_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with launcher.setup_lock():
        with pytest.raises(RuntimeError, match="Another dashboard setup"):
            with launcher.setup_lock():
                pass
