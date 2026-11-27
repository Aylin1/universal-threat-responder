import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"),
    reason="The Bash launcher targets Linux and macOS",
)


@pytest.fixture
def commands(tmp_path):
    scripts = {
        "uname": 'printf "%s\\n" "$TEST_PLATFORM"',
        "docker": (
            'printf "docker %s\\n" "$*" >> "$TEST_LOG"\n'
            'if [ "$1" = info ]; then printf "%s\\n" "$TEST_DOCKER_OS"; exit 0; fi\n'
            'printf "address=%s\\n" "${DASHBOARD_OLLAMA_ADDRESS:-default}" >> "$TEST_LOG"\n'
            'exit "${TEST_DOCKER_EXIT:-0}"'
        ),
        "xdg-open": 'printf "xdg-open %s\\n" "$*" >> "$TEST_LOG"',
        "open": 'printf "open %s\\n" "$*" >> "$TEST_LOG"',
    }
    for name, body in scripts.items():
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
    return tmp_path


def run_launcher(commands, **values):
    log = commands / "calls"
    env = os.environ.copy()
    env.update({
        "PATH": f"{commands}{os.pathsep}{env['PATH']}",
        "TEST_LOG": str(log),
        "TEST_PLATFORM": "Linux",
        "TEST_DOCKER_OS": "Docker Engine",
        "DASHBOARD_PORT": "8503",
        **values,
    })
    env.pop("DASHBOARD_OLLAMA_ADDRESS", None)
    result = subprocess.run(
        ["bash", str(ROOT / "start-dashboard-docker.sh")],
        env=env, capture_output=True, text=True,
    )
    return result, log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("platform", "docker_os", "address", "browser"),
    [
        ("Linux", "Docker Engine", "docker-host-gateway", "xdg-open"),
        ("Linux", "Docker Desktop", "default", "xdg-open"),
        ("Darwin", "", "default", "open"),
    ],
)
def test_starts_dashboard_and_opens_browser(commands, platform, docker_os, address, browser):
    result, calls = run_launcher(commands, TEST_PLATFORM=platform, TEST_DOCKER_OS=docker_os)
    assert result.returncode == 0, result.stderr
    assert "docker compose up --wait dashboard\n" in calls
    assert f"address={address}\n" in calls
    assert f"{browser} http://localhost:8503\n" in calls


def test_does_not_open_browser_if_dashboard_fails(commands):
    result, calls = run_launcher(commands, TEST_DOCKER_EXIT="7")
    assert result.returncode == 7
    assert "xdg-open" not in calls
