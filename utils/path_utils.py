from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
WSL_MOUNT_RE = re.compile(r"^/mnt/(?P<drive>[a-zA-Z])/(?P<rest>.*)$")


def windows_to_wsl_path(path: str) -> str:
    match = WINDOWS_DRIVE_RE.match(path or "")
    if not match:
        return path
    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def wsl_to_windows_path(path: str) -> str:
    match = WSL_MOUNT_RE.match(path or "")
    if not match:
        return path
    drive = match.group("drive").upper()
    rest = match.group("rest").replace("/", "\\")
    return f"{drive}:\\{rest}"


def normalize_local_path(path: str) -> str:
    if not path:
        return ""

    candidate = path.strip().strip('"').strip("'")
    if WINDOWS_DRIVE_RE.match(candidate):
        wsl_candidate = windows_to_wsl_path(candidate)
        return os.path.abspath(wsl_candidate if os.path.exists(wsl_candidate) else wsl_candidate)

    expanded = os.path.abspath(os.path.expanduser(candidate))
    return expanded


def ensure_display_path(path: str) -> str:
    if not path:
        return ""
    normalized = normalize_local_path(path)
    windows_variant = wsl_to_windows_path(normalized)
    return windows_variant if windows_variant != normalized else normalized


def open_in_file_manager(path: str) -> None:
    normalized = normalize_local_path(path)
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"文件不存在: {path}")

    windows_path = wsl_to_windows_path(normalized)
    if windows_path != normalized:
        subprocess.run(["explorer.exe", f"/select,{windows_path}"], check=False, timeout=10)
        return

    if os.name == "nt":
        subprocess.run(["explorer", f"/select,{normalized}"], check=False, timeout=10)
        return

    target_dir = str(Path(normalized).parent)
    subprocess.run(["xdg-open", target_dir], check=False, timeout=10)


def same_file_path(left: str, right: str) -> bool:
    normalized_left = normalize_local_path(left)
    normalized_right = normalize_local_path(right)
    return os.path.normcase(normalized_left) == os.path.normcase(normalized_right)
