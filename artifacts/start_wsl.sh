#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
REQUIREMENTS_HASH_FILE="${VENV_DIR}/.requirements.sha256"
DATA_DIR="${PROJECT_ROOT}/data"
RUNTIME_DIR="${PROJECT_ROOT}/artifacts/runtime-wsl"
INDEX_PATH="${DATA_DIR}/photo_search.index"
METADATA_PATH="${DATA_DIR}/metadata.json"

ensure_directory() {
  local path="$1"
  mkdir -p "${path}"
}

get_dotenv_value() {
  local file_path="$1"
  local key="$2"
  if [[ ! -f "${file_path}" ]]; then
    return 1
  fi

  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" =~ ^${key}[[:space:]]*=[[:space:]]*(.*)$ ]]; then
      local value="${BASH_REMATCH[1]}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ "${value}" =~ ^\"(.*)\"$ || "${value}" =~ ^\'(.*)\'$ ]]; then
        value="${BASH_REMATCH[1]}"
      fi
      printf '%s\n' "${value}"
      return 0
    fi
  done < "${file_path}"

  return 1
}

convert_windows_to_wsl_path() {
  local input_path="$1"
  if [[ "${input_path}" =~ ^([A-Za-z]):[\\/](.*)$ ]]; then
    local drive="${BASH_REMATCH[1],,}"
    local rest="${BASH_REMATCH[2]//\\//}"
    printf '/mnt/%s/%s\n' "${drive}" "${rest}"
    return 0
  fi
  printf '%s\n' "${input_path}"
}

resolve_absolute_path() {
  local input_path="$1"
  local base_dir="$2"
  if [[ -z "${input_path}" ]]; then
    return 1
  fi
  if [[ "${input_path}" = /* ]]; then
    realpath -m "${input_path}"
    return 0
  fi
  realpath -m "${base_dir}/${input_path}"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "[STEP] Installing uv for WSL..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installation failed in WSL." >&2
    exit 1
  fi
}

ensure_wsl_python_environment() {
  ensure_uv

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[STEP] Creating WSL virtual environment at ${VENV_DIR}"
    uv venv "${VENV_DIR}" --python 3.12
  fi

  local requirements_hash
  requirements_hash="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
  local installed_hash=""
  if [[ -f "${REQUIREMENTS_HASH_FILE}" ]]; then
    installed_hash="$(tr -d '[:space:]' < "${REQUIREMENTS_HASH_FILE}")"
  fi

  if [[ "${installed_hash}" != "${requirements_hash}" ]]; then
    echo "[STEP] Installing Python dependencies for WSL runtime"
    uv pip install --python "${PYTHON_BIN}" -r "${REQUIREMENTS_FILE}"
    printf '%s\n' "${requirements_hash}" > "${REQUIREMENTS_HASH_FILE}"
  fi
}

port_is_busy() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q .
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -P -n >/dev/null 2>&1
    return $?
  fi
  return 1
}

ensure_directory "${DATA_DIR}"
ensure_directory "${RUNTIME_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${ENV_EXAMPLE_FILE}" ]]; then
    cp "${ENV_EXAMPLE_FILE}" "${ENV_FILE}"
  fi
  echo "Missing .env. A new .env has been created from .env.example. Please set PHOTO_DIR and your API settings, then rerun the script." >&2
  exit 1
fi

RAW_PHOTO_DIR="${PHOTO_DIR:-}"
if [[ -z "${RAW_PHOTO_DIR}" ]]; then
  RAW_PHOTO_DIR="$(get_dotenv_value "${ENV_FILE}" "PHOTO_DIR" || true)"
fi
if [[ -z "${RAW_PHOTO_DIR}" ]]; then
  echo "PHOTO_DIR is not set. Please update .env before running this script." >&2
  exit 1
fi

PHOTO_DIR_VALUE="$(convert_windows_to_wsl_path "${RAW_PHOTO_DIR}")"
PHOTO_DIR_VALUE="$(resolve_absolute_path "${PHOTO_DIR_VALUE}" "${PROJECT_ROOT}")"
if [[ ! -d "${PHOTO_DIR_VALUE}" ]]; then
  echo "PHOTO_DIR does not exist in WSL: ${PHOTO_DIR_VALUE}" >&2
  exit 1
fi

SERVER_PORT_VALUE="${1:-}"
if [[ -z "${SERVER_PORT_VALUE}" ]]; then
  SERVER_PORT_VALUE="${SERVER_PORT:-}"
fi
if [[ -z "${SERVER_PORT_VALUE}" ]]; then
  SERVER_PORT_VALUE="$(get_dotenv_value "${ENV_FILE}" "SERVER_PORT" || true)"
fi
if [[ -z "${SERVER_PORT_VALUE}" ]]; then
  SERVER_PORT_VALUE="10001"
fi

if port_is_busy "${SERVER_PORT_VALUE}"; then
  echo "Port ${SERVER_PORT_VALUE} is already in use inside WSL. Stop the existing process or rerun with a different port." >&2
  exit 1
fi

ensure_wsl_python_environment

export PHOTO_DIR="${PHOTO_DIR_VALUE}"
export DATA_DIR="${DATA_DIR}"
export RUNTIME_DATA_DIR="${DATA_DIR}"
export INDEX_PATH="${INDEX_PATH}"
export METADATA_PATH="${METADATA_PATH}"
export SERVER_HOST="127.0.0.1"
export SERVER_PORT="${SERVER_PORT_VALUE}"

echo
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Project name: ${PROJECT_NAME}"
echo "[INFO] WSL runtime Python: ${PYTHON_BIN}"
echo "[INFO] PHOTO_DIR: ${PHOTO_DIR}"
echo "[INFO] DATA_DIR: ${DATA_DIR}"
echo "[INFO] INDEX_PATH: ${INDEX_PATH}"
echo "[INFO] METADATA_PATH: ${METADATA_PATH}"
echo "[INFO] App URL: http://127.0.0.1:${SERVER_PORT}/"
echo
echo "[INFO] This launcher runs only inside WSL and does not call PowerShell."
echo

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py"
