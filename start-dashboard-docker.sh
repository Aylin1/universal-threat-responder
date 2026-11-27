#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

platform=$(uname -s)
case "$platform" in
    Linux|Darwin) ;;
    *)
        echo "ERROR: This launcher supports Linux and macOS." >&2
        exit 1
        ;;
esac

if [[ "$platform" == Linux ]]; then
    docker_os=$(docker info --format '{{.OperatingSystem}}')
    if [[ "$docker_os" != *"Docker Desktop"* ]]; then
        export DASHBOARD_OLLAMA_ADDRESS="${DASHBOARD_OLLAMA_ADDRESS:-docker-host-gateway}"
    fi
fi

echo "Starting the dashboard using your host Ollama. First-time setup may take several minutes."
docker compose up --wait dashboard

url="http://localhost:${DASHBOARD_PORT:-8501}"
echo "Dashboard ready: $url"
if [[ "$platform" == Darwin ]]; then
    if ! open "$url"; then
        echo "Could not open a browser; open $url manually." >&2
        exit 1
    fi
elif command -v xdg-open >/dev/null 2>&1; then
    if ! xdg-open "$url"; then
        echo "Could not open a browser; open $url manually." >&2
        exit 1
    fi
else
    echo "Could not open a browser automatically; open $url manually." >&2
fi
