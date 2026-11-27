"""Prepare and launch the Windows dashboard."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import venv

ROOT = Path(__file__).resolve().parent.parent


def select_environment(root):
    for name in (".dashboard-venv", ".venv"):
        environment = root / name
        if environment.exists():
            return environment
    return root / ".dashboard-venv"


VENV = select_environment(ROOT)
PYTHON = Path(sys.executable) if os.environ.get("DASHBOARD_CONTAINER") == "1" else VENV / "Scripts" / "python.exe"
MARKER = VENV / "dashboard-requirements.sha256"
PINNED = ("scikit-learn==1.9.0", "xgboost==3.2.0")
STATUS_INTERVAL_SECONDS = 30


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


@contextmanager
def waiting_for(stage):
    started = time.monotonic()
    stopped = threading.Event()
    print(f"{stage}... Please keep this window open; the time required varies.", flush=True)

    def report_wait():
        while not stopped.wait(STATUS_INTERVAL_SECONDS):
            elapsed = int(time.monotonic() - started)
            print(f"Still waiting for {stage} ({elapsed}s elapsed). Please keep this window open.", flush=True)

    reporter = threading.Thread(target=report_wait, daemon=True)
    reporter.start()
    try:
        yield
    finally:
        stopped.set()
        reporter.join()
    print(f"Finished: {stage}.", flush=True)


def prepare_environment():
    if not VENV.exists():
        print("Creating the dashboard Python environment...", flush=True)
        venv.create(VENV, with_pip=True)
    else:
        print(f"Reusing Python environment: {VENV}", flush=True)
    if not PYTHON.is_file():
        raise RuntimeError(f"Incomplete dashboard environment at {VENV}; repair or remove it before retrying.")
    run([
        str(PYTHON), "-c",
        "import sys; sys.version_info[:2] == (3, 11) or sys.exit('The dashboard environment must use Python 3.11.')",
    ])
    digest = hashlib.sha256((ROOT / "requirements.txt").read_bytes() + "\n".join(PINNED).encode()).hexdigest()
    if not MARKER.exists() or MARKER.read_text(encoding="utf-8").strip() != digest:
        print("Installing dashboard dependencies; this may take several minutes...", flush=True)
        run([str(PYTHON), "-m", "pip", "install", "-r", "requirements.txt", *PINNED])
        run([str(PYTHON), "-m", "pip", "check"])
        MARKER.write_text(digest, encoding="utf-8")
    else:
        run([str(PYTHON), "-m", "pip", "check"])


def prepare_ollama():
    from config import cfg

    ollama = shutil.which("ollama")
    if not ollama:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        ollama = str(candidate) if candidate.is_file() else None
    if not ollama:
        raise RuntimeError("Ollama was not found. Install Ollama, then restart the launcher.")
    result = subprocess.run([ollama, "list"], capture_output=True, text=True)
    if result.returncode:
        if os.environ.get("OLLAMA_HOST"):
            raise RuntimeError(f"Cannot connect to configured OLLAMA_HOST: {result.stderr.strip()}")
        print("Starting Ollama...", flush=True)
        log_path = ROOT / "logs" / "ollama-launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [ollama, "serve"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
            )
        for _ in range(30):
            time.sleep(1)
            result = subprocess.run([ollama, "list"], capture_output=True, text=True)
            if result.returncode == 0:
                break
            if process.poll() is not None:
                break
        if result.returncode:
            raise RuntimeError(f"Ollama did not start: {result.stderr.strip()}. See {log_path}")
    names = {line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()}
    model = cfg.llm.model_name
    default_tag = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    if model not in names and default_tag not in names:
        print(f"Downloading Ollama model {model}; this may take a long time and several GB...", flush=True)
        run([ollama, "pull", model])


def prepare_container_ollama():
    import requests
    from config import cfg

    host = os.environ["OLLAMA_HOST"].rstrip("/")
    model = cfg.llm.model_name
    default_tag = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        response.raise_for_status()
        names = {entry["name"] for entry in response.json()["models"]}
        if model not in names and default_tag not in names:
            print(f"Downloading Ollama model {model}; this may take a long time and several GB...", flush=True)
            response = requests.post(
                f"{host}/api/pull", json={"model": model, "stream": False}, timeout=(10, None),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") != "success":
                raise RuntimeError(f"Ollama model download failed: {result}")
    except requests.RequestException as error:
        raise RuntimeError(
            f"Cannot access Ollama at {host}. Check the Ollama service and Docker networking: {error}"
        ) from error


def prepare_classifier():
    from adapters.phishfuzzer_loader import load_classifier
    from config import cfg

    if not Path(cfg.dataset.classifier_path).is_file():
        raise RuntimeError(
            f"Saved classifier not found: {cfg.dataset.classifier_path}. "
            "Restore the trusted classifier or build it separately with python -m scripts.build_classifier."
        )
    load_classifier(cfg.dataset.classifier_path)


@contextmanager
def setup_lock():
    import msvcrt

    with (ROOT / ".dashboard-setup.lock").open("a+b") as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another dashboard setup is running. Wait for it to finish and retry.") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    if sys.argv[1:] == ["--container"]:
        os.chdir(ROOT)
        with waiting_for("Checking Ollama and its model"):
            prepare_container_ollama()
        with waiting_for("Checking the saved classifier"):
            run([str(PYTHON), "-m", "scripts.launch_dashboard", "--check-classifier"])
        print("Starting Streamlit; open the host port configured in Docker Compose.", flush=True)
        run([
            str(PYTHON), "-m", "streamlit", "run", "app.py",
            "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ])
        return
    if sys.argv[1:] == ["--check-classifier"]:
        prepare_classifier()
        return
    if sys.argv[1:]:
        raise ValueError(f"Unknown arguments: {sys.argv[1:]}")
    os.chdir(ROOT)
    with setup_lock():
        with waiting_for("Preparing Python and dependencies"):
            prepare_environment()
        with waiting_for("Preparing Ollama and its model"):
            prepare_ollama()
        with waiting_for("Checking the saved classifier"):
            run([str(PYTHON), "-m", "scripts.launch_dashboard", "--check-classifier"])
    print("Starting Streamlit. Initializing the dashboard may take a few minutes. "
          "Keep this window open while using it.", flush=True)
    run([str(PYTHON), "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
