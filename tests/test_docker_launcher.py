import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="The Docker launcher is a Windows batch file")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fake_commands(tmp_path_factory):
    directory = tmp_path_factory.mktemp("docker-launcher-commands")
    binary = directory / "docker.exe"
    source = r"""
using System;
using System.IO;
public class FakeCommand {
    public static int Main(string[] args) {
        if (args.Length > 1 && args[0] == "compose" && args[1] == "up")
            File.WriteAllText(Environment.GetEnvironmentVariable("FAKE_DOCKER_STARTED"),
                              string.Join(" ", args));
        return 0;
    }
}
"""
    script = "$source = @'\n" + source + "\n'@\nAdd-Type -TypeDefinition $source -OutputAssembly $env:FAKE_BINARY -OutputType ConsoleApplication"
    env = os.environ.copy()
    env["FAKE_BINARY"] = str(binary)
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], env=env, check=True, capture_output=True)
    return directory


def test_docker_launcher_starts_compose_without_host_ollama(tmp_path, fake_commands):
    launcher = tmp_path / "start-dashboard-docker.cmd"
    shutil.copyfile(ROOT / "start-dashboard-docker.cmd", launcher)
    docker_started = tmp_path / "docker-started"

    env = os.environ.copy()
    env.update({
        "PATH": str(fake_commands) + os.pathsep + env["PATH"],
        "FAKE_DOCKER_STARTED": str(docker_started),
        "OLLAMA_HOST": "remote.example:11434",
    })
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher)], env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert docker_started.read_text() == "compose up --build"
