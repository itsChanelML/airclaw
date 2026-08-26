#!/usr/bin/env python3
"""
AirClaw pre-show check
----------------------
Run this before you walk on stage. It verifies, in about ten seconds, every
external thing the demo depends on and cannot control.

    python3 preflight.py              # full check
    python3 preflight.py --no-airflow # skip the DAG parse (faster)

Why it exists: on 2026-08-26 the model this project ran on reached its NVIDIA
end-of-life at 09:00 UTC. Every NIM call started returning HTTP 410 Gone, and
the failure looked like an API key problem. A live tool-call against the model
is the only way to catch that class of failure before an audience does.

Exit code 0 means you are clear to demo. Non-zero means read the output.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

GREEN = "\033[38;5;82m"
RED   = "\033[38;5;196m"
AMBER = "\033[38;5;214m"
GRAY  = "\033[38;5;245m"
TEAL  = "\033[38;5;43m"
BOLD  = "\033[1m"
RESET = "\033[0m"

failures = []
warnings = []


def ok(label, detail=""):
    print(f"  {GREEN}PASS{RESET}  {label}" + (f"  {GRAY}{detail}{RESET}" if detail else ""))


def fail(label, detail=""):
    failures.append(label)
    print(f"  {RED}{BOLD}FAIL{RESET}  {label}" + (f"  {GRAY}{detail}{RESET}" if detail else ""))


def warn(label, detail=""):
    warnings.append(label)
    print(f"  {AMBER}WARN{RESET}  {label}" + (f"  {GRAY}{detail}{RESET}" if detail else ""))


def section(title):
    print(f"\n{TEAL}{title}{RESET}")


# ── 1. Dependencies ────────────────────────────────────────────────────────────

def check_dependencies():
    section("Dependencies")
    for module, why in (("requests", "NIM API calls"),
                        ("pydantic", "tool schemas"),
                        ("dotenv",   ".env loading")):
        try:
            __import__(module)
            ok(module, why)
        except ImportError:
            fail(module, f"pip3 install -r requirements.txt  ({why})")


# ── 2. API key ─────────────────────────────────────────────────────────────────

def check_key():
    section("API key")
    from airclaw_env import get_nim_key
    key, error = get_nim_key()
    if error:
        fail("NIM_API_KEY", error.splitlines()[0])
        return None
    ok("NIM_API_KEY", f"loaded, {len(key)} chars, nvapi- prefix")
    return key


# ── 3. The model itself ────────────────────────────────────────────────────────

def check_model(key):
    section("Model")
    if not key:
        fail("model reachability", "skipped — no usable API key")
        return

    import requests
    from airclaw_env import get_model
    model = get_model()

    probe_tool = [{
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Reply that you are reachable.",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        },
    }]

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model":       model,
                "messages":    [{"role": "user", "content": "Call the ping tool with ok=true."}],
                "tools":       probe_tool,
                "tool_choice": "auto",
                "max_tokens":  64,
                "temperature": 0,
            },
            timeout=90,
        )
    except Exception as e:
        fail(f"{model} unreachable", f"{type(e).__name__}: {e}")
        return

    if response.status_code == 410:
        fail(f"{model} is RETIRED",
             "NVIDIA end-of-life. Set NIM_MODEL in .env to a served model. "
             "List them: curl -s https://integrate.api.nvidia.com/v1/models "
             '-H "Authorization: Bearer $NIM_API_KEY" | grep -o \'"id":"[^"]*"\'')
        return
    if response.status_code == 401:
        fail(f"{model} rejected the key", "401 Unauthorized — check .env")
        return
    if response.status_code != 200:
        fail(f"{model} returned {response.status_code}", response.text[:160])
        return

    message    = response.json()["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        ok(model, "reachable, tool calling works")
    else:
        # The agent architecture is entirely tool calls. Prose here means the
        # pipeline will stall and fall back on every run.
        warn(model, "reachable but did not emit a tool call on a trivial prompt")


# ── 4. Data ────────────────────────────────────────────────────────────────────

def check_data():
    section("Data")
    expected = {
        "nyc_311_clean.csv":       ("sla_breach", 300),
        "nyc_311_broken.csv":      ("sla_breach", 300),
        "model_eval_clean.csv":    ("model_b_quality_score", 300),
        "model_eval_broken.csv":   ("model_b_score", 300),
    }
    for name, (column, rows_expected) in expected.items():
        path = ROOT / "data" / name
        if not path.exists():
            fail(name, "missing")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            count  = sum(1 for _ in reader)
        if column not in fields:
            fail(name, f"expected column '{column}' not found")
        elif count != rows_expected:
            warn(name, f"{count} rows, expected {rows_expected}")
        else:
            ok(name, f"{count} rows")


def check_rebasing():
    section("Date rebasing")
    from rebase_data import describe_shift, rebase_csv
    import airclaw_tools as tools

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "rebased.csv"
        try:
            shift = rebase_csv(ROOT / "data" / "nyc_311_clean.csv", target)
        except AssertionError as e:
            fail("rebasing invariants", str(e)[:160])
            return
        ok("rebasing invariants", f"shift {describe_shift(shift)}, breach counts preserved")

        breaches = tools.check_sla_breaches(tools.SLABreachInput(file_path=str(target)))
        total    = breaches.data.get("total", 0)
        if total == 29:
            ok("SLA breaches", "29 — matches the talk track")
        else:
            fail("SLA breaches", f"{total}, expected 29 (talk track says 29)")

        spikes = tools.detect_complaint_spike(tools.SpikeDetectionInput(file_path=str(target)))
        found  = spikes.data.get("spikes", [])
        if found:
            top = found[0]
            ok("complaint spikes",
               f"{len(found)} detected, largest {top['complaint_type']} "
               f"{top['increase_pct']}")
        else:
            fail("complaint spikes", "none detected — the spike beat will not fire")


# ── 5. Tool registries ─────────────────────────────────────────────────────────

def check_registries():
    section("Tool registries")
    import airclaw_tools, model_eval_tools
    for label, module in (("311 triage", airclaw_tools), ("model eval", model_eval_tools)):
        registry = set(module.TOOL_REGISTRY)
        schemas  = {s["function"]["name"] for s in module.TOOL_SCHEMAS}
        if registry == schemas:
            ok(label, f"{len(registry)} tools, registry matches schemas")
        else:
            fail(label,
                 f"registry/schema mismatch — only in registry: {registry - schemas}, "
                 f"only in schemas: {schemas - registry}")


# ── 6. Airflow DAGs ────────────────────────────────────────────────────────────

def check_airflow():
    section("Airflow DAGs")
    script = ROOT / "run_airflow.sh"
    if not script.exists():
        warn("run_airflow.sh", "not found — skipping DAG parse")
        return
    try:
        result = subprocess.run(
            ["bash", str(script), "--check"],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        warn("DAG parse", "timed out after 180s")
        return
    if result.returncode == 0 and "parsed cleanly" in result.stdout:
        ok("both DAGs", "parse cleanly under Airflow 3")
    else:
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        fail("DAG parse", " | ".join(line.strip() for line in tail))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AirClaw pre-show check")
    parser.add_argument("--no-airflow", action="store_true",
                        help="skip the Airflow DAG parse check")
    args = parser.parse_args()

    print(f"\n{BOLD}{TEAL}AirClaw — pre-show check{RESET}")

    check_dependencies()
    key = check_key()
    check_model(key)
    check_data()
    check_rebasing()
    check_registries()
    if not args.no_airflow:
        check_airflow()

    print()
    if failures:
        print(f"{RED}{BOLD}NOT READY — {len(failures)} check(s) failed:{RESET}")
        for label in failures:
            print(f"  {RED}• {label}{RESET}")
        print(f"\n{GRAY}Fix these before going on stage.{RESET}\n")
        return 1

    if warnings:
        print(f"{AMBER}{BOLD}READY, with {len(warnings)} warning(s).{RESET}")
        for label in warnings:
            print(f"  {AMBER}• {label}{RESET}")
        print()
        return 0

    print(f"{GREEN}{BOLD}READY — all checks passed. Go do the thing.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
