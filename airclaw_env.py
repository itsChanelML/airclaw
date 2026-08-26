"""
AirClaw Environment Loader
--------------------------
Single place that loads .env and validates the NIM API key.

Used by run_demo.py, run_model_eval.py, and the NemoClawOperator so all
three behave identically — the docs promise .env is loaded automatically,
and this is what makes that true.

Python 3.9 compatible.
"""

import os
from pathlib import Path

ROOT     = Path(__file__).parent
ENV_FILE = ROOT / ".env"

# The placeholder shipped in .env.example. If someone copies the example and
# forgets to fill it in, we catch it here instead of sending
# "Bearer your_nim_api_key_here" to NIM and surfacing a confusing 401.
PLACEHOLDER = "your_nim_api_key_here"


def load_env() -> None:
    """Load .env into os.environ if python-dotenv is available. Never raises."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


# NVIDIA retires hosted models on a published EOL date, and when that happens
# every call returns HTTP 410 Gone. llama-3.3-nemotron-super-49b-v1 — the model
# this project launched on — reached end of life on 2026-08-26T09:00Z.
# nemotron-3-super-120b-a12b is its successor in the same Super line.
# Override with NIM_MODEL in .env if you want a different one.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def get_model() -> str:
    """Resolve the NIM model id — NIM_MODEL wins, otherwise the default."""
    load_env()
    return (os.environ.get("NIM_MODEL") or "").strip() or DEFAULT_MODEL


def get_nim_key(env_var: str = "NIM_API_KEY"):
    """
    Return (api_key, error_message).

    On success: (key, None). On failure: (None, human-readable reason).
    Callers decide how to fail — the runners print and exit, the Airflow
    operator raises.
    """
    load_env()
    key = (os.environ.get(env_var) or "").strip()

    if not key:
        return None, (
            f"{env_var} is not set.\n"
            f"  Fix: add it to {ENV_FILE} as {env_var}=nvapi-...\n"
            f"  Or:  export {env_var}=nvapi-...\n"
            f"  Get a key free at https://build.nvidia.com"
        )

    if key == PLACEHOLDER or key.startswith("your_"):
        return None, (
            f"{env_var} is still the placeholder value from .env.example.\n"
            f"  Fix: open {ENV_FILE} and replace it with your real key.\n"
            f"  Get a key free at https://build.nvidia.com"
        )

    if not key.startswith("nvapi-"):
        return None, (
            f"{env_var} does not look like an NVIDIA NIM key "
            f"(expected it to start with 'nvapi-', got '{key[:6]}…').\n"
            f"  Check you copied the whole key from https://build.nvidia.com"
        )

    return key, None


# ── Path resolution ────────────────────────────────────────────────────────────
# The Airflow path used to break because DAGs resolved data files as
# "os.path.dirname(__file__)/../data" — correct when running from the repo,
# wrong the moment the DAG is copied into AIRFLOW_HOME. These helpers work
# either way: explicit AIRCLAW_HOME wins, otherwise walk up looking for the
# repo layout, otherwise fall back to this file's directory.

def repo_root() -> Path:
    """Locate the AirClaw repo root regardless of how the caller was invoked."""
    explicit = os.environ.get("AIRCLAW_HOME")
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate

    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (base, *base.parents):
            if (candidate / "data").is_dir() and (candidate / "tools").is_dir():
                return candidate

    return ROOT.resolve()


def data_file(name: str) -> Path:
    """Absolute path to a file in the repo's data/ directory."""
    return repo_root() / "data" / name


def tools_dir() -> Path:
    """Absolute path to the repo's tools/ directory (for sys.path insertion)."""
    return repo_root() / "tools"
