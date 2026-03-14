#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERSHELL_EXE="${POWERSHELL_EXE:-/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe}"
WINDOWS_SCRIPT_PATH="$(wslpath -w "${SCRIPT_DIR}/start_stack.ps1")"

if [[ ! -x "${POWERSHELL_EXE}" ]]; then
  echo "PowerShell executable not found: ${POWERSHELL_EXE}" >&2
  exit 1
fi

exec "${POWERSHELL_EXE}" -NoProfile -ExecutionPolicy Bypass -File "${WINDOWS_SCRIPT_PATH}"
