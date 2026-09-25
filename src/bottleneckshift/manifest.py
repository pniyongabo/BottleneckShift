"""Best-effort environment capture without making measurement depend on tooling."""

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _capture(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def environment(repo: Path) -> dict:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _capture(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_status": _capture(["git", "-C", str(repo), "status", "--short"]),
        "packages": _capture([sys.executable, "-m", "pip", "freeze", "--all"]),
        "gpu_inventory": _capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
    }
