#!/usr/bin/env python3
"""
AirClaw Model Eval Runner — GPT-4o vs Nemotron-Super
-----------------------------------------------------
python3 run_model_eval.py               # happy path
python3 run_model_eval.py --break       # ESCALATE beat
export NIM_API_KEY=your_key
"""

import argparse, inspect, json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
TOOLS_DIR = ROOT / "tools"
UPSTREAM  = DATA_DIR / "model_eval_upstream.csv"
CLEAN     = DATA_DIR / "model_eval_clean.csv"
BROKEN    = DATA_DIR / "model_eval_broken.csv"

sys.path.insert(0, str(TOOLS_DIR))
import model_eval_tools as tools_mod
from model_eval_tools import (
    TOOL_REGISTRY, TOOL_SCHEMAS, AgentStatus, trim_for_history,
)

sys.path.insert(0, str(ROOT))
from airclaw_env import get_model, get_nim_key

import requests

TEAL   = "\033[38;5;43m"
PURPLE = "\033[38;5;135m"
AMBER  = "\033[38;5;214m"
GREEN  = "\033[38;5;82m"
RED    = "\033[38;5;196m"
GRAY   = "\033[38;5;245m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

NIM_MODEL    = get_model()
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_ITER     = 10

REQUIRED_FIELDS = [
    "prompt_id", "category", "model_a", "model_b",
    "model_a_quality_score", "model_b_quality_score",
    "model_a_format_score", "model_b_format_score",
    "model_a_latency_sec", "model_b_latency_sec",
    "model_a_cost_usd", "model_b_cost_usd",
    "model_a_refused", "model_b_refused", "regression_flag",
]

MONTHLY_VOLUME = 100_000
TOTAL_PROMPTS  = 300


def eval_models(path):
    """
    Read the two models under evaluation from the eval file itself.

    The model the agent RUNS ON (NIM_MODEL) and the models it is EVALUATING are
    different things. Conflating them meant the report could name one model in
    the cost analysis and a different one in the summary.
    """
    import csv as _csv
    try:
        with open(path, newline="") as f:
            row = next(_csv.DictReader(f))
        return row.get("model_a", "Model A"), row.get("model_b", "Model B")
    except Exception:
        return "Model A", "Model B"

SYSTEM_PROMPT = (
    "You are NemoClaw. Call tools one at a time. Never write text between tool calls. "
    "After each tool result, immediately call the next tool. "
    "Do NOT write tool calls as text. Execute them directly."
)

def banner(t):
    w = 62
    print(f"\n{BOLD}{TEAL}{'─'*w}{RESET}")
    print(f"{BOLD}{TEAL}  {t}{RESET}")
    print(f"{BOLD}{TEAL}{'─'*w}{RESET}")

def div():      print(f"{GRAY}{'·'*52}{RESET}")
def log_t(m):   print(f"{TEAL}{m}{RESET}")
def log_p(m):   print(f"{PURPLE}{m}{RESET}")
def log_g(m):   print(f"{GREEN}{m}{RESET}")
def log_a(m):   print(f"{AMBER}{m}{RESET}")
def log_r(m):   print(f"{RED}{BOLD}{m}{RESET}")
def log_gr(m):  print(f"{GRAY}{m}{RESET}")

def log_status(s):
    c = GREEN if s == "SUCCESS" else (AMBER if s == "RETRY" else RED)
    print(f"{c}{BOLD}  Status        : {s}{RESET}")

def call_nim(api_key, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       NIM_MODEL,
        "messages":    messages,
        "tools":       TOOL_SCHEMAS,
        "tool_choice": "auto",
        "max_tokens":  1024,
        "temperature": 0.1,
    }
    timeouts = [120, 150, 180]
    for attempt, timeout in enumerate(timeouts, 1):
        try:
            r = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < len(timeouts):
                log_a(f"NIM timeout (attempt {attempt}/{len(timeouts)}) — retrying with {timeouts[attempt]}s timeout…")
                time.sleep(3)
            else:
                log_r(f"NIM error: timed out after {timeout}s")
                return None
        except Exception as e:
            log_r(f"NIM error: {e}")
            return None

def invoke_tool(fn, args):
    sig    = inspect.signature(fn)
    params = list(sig.parameters.values())
    if params:
        model = params[0].annotation
        try:
            return fn(model(**args))
        except Exception as e:
            return tools_mod.AgentResult(
                status=AgentStatus.RETRY,
                message=f"Input error: {e}",
                tool=fn.__name__
            )
    return fn(args)

def _deterministic_fallback(missing):
    """
    Safety net for the live demo, NOT the happy path.

    The agent is expected to call draft_migration_report and generate_summary
    itself — the tools are self-hydrating now, so it only needs file_path and
    the two model names. If the model stalls anyway (NIM hiccup, dropped tool
    call), we still finish the run rather than dying on stage.

    Every line this prints is labeled [fallback] so the logs never claim the
    agent did something it did not do.
    """
    from model_eval_tools import (
        draft_migration_report, generate_summary,
        DraftMigrationReportInput, GenerateSummaryInput
    )

    model_a, model_b = eval_models(str(UPSTREAM.resolve()))
    log_a(f"[fallback] Agent did not call: {', '.join(missing)}")
    log_a("[fallback] Completing these steps deterministically — not agent output.")
    div()

    rec = ""
    if "draft_migration_report" in missing:
        log_a(f"[fallback] Running tool : {BOLD}draft_migration_report{RESET}{AMBER}")
        r = draft_migration_report(DraftMigrationReportInput(
            model_a=model_a,
            model_b=model_b,
            file_path=str(UPSTREAM.resolve()),
            monthly_volume=MONTHLY_VOLUME,
        ))
        log_status(r.status.value)
        log_gr(f"  Message       : {r.message[:220]}")
        stream_report(r)
        rec = r.data.get("recommendation", "")

    if "generate_summary" in missing:
        log_a(f"[fallback] Running tool : {BOLD}generate_summary{RESET}{AMBER}")
        s_res = generate_summary(GenerateSummaryInput(
            recommendation=rec,
            model_a=model_a,
            model_b=model_b,
            total_evaluated=TOTAL_PROMPTS,
        ))
        log_status(s_res.status.value)
        log_gr(f"  Message       : {s_res.message[:220]}")
        div()
        return s_res

    return None


def stream_report(result):
    """Print the migration report body — the beat where you stop talking."""
    if not result.data.get("report"):
        return
    div()
    log_t("  MIGRATION REPORT:")
    print()
    for line in result.data["report"].split("\n"):
        print(f"  {GRAY}{line}{RESET}")
    print()
    div()


def run_agent(api_key):
    banner("NemoClaw Eval Agent — Model Migration Analysis Starting")
    model_a, model_b = eval_models(str(UPSTREAM.resolve()))
    log_t(f"Comparing : {model_a}  ->  {model_b}")
    log_t(f"Agent runs on : {NIM_MODEL}")
    log_t(f"Prompts   : {TOTAL_PROMPTS} production samples across 4 task categories")
    log_t(f"As of     : {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    div()

    fp         = str(UPSTREAM.resolve())
    req_fields = json.dumps(REQUIRED_FIELDS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Analyze a production model migration from {model_a} to {model_b}.\n\n"
            f"Eval file: {fp}\n\n"
            f"Call validate_schema with these EXACT required_fields — do not change them:\n"
            f"{req_fields}\n\n"
            "Then call these tools, one per turn, in this order:\n"
            "2. score_comparison  — file_path only\n"
            "3. detect_regression — file_path only\n"
            f"4. cost_analysis     — file_path, monthly_volume={MONTHLY_VOLUME}\n"
            f"5. draft_migration_report — model_a='{model_a}', model_b='{model_b}', "
            f"file_path=<same path>, monthly_volume={MONTHLY_VOLUME}. "
            "It reads the eval file itself, so do NOT pass the earlier tool payloads.\n"
            "6. generate_summary  — pass the recommendation string from step 5, "
            f"model_a, model_b, and total_evaluated.\n\n"
            "Start now. Call validate_schema."
        )},
    ]

    called       = []
    final_result = None

    for iteration in range(1, MAX_ITER + 1):
        log_p(f"[Iteration {iteration}] Calling NemoClaw…")

        response = call_nim(api_key, messages)
        if response is None:
            log_r("NIM call failed — retrying in 2s…")
            time.sleep(2)
            continue

        assistant_msg = response["choices"][0]["message"]
        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            thought = assistant_msg.get("content", "")
            log_g("Agent completed reasoning — no further tool calls.")
            if thought:
                log_gr(f"Final thought: {thought[:200]}")
            div()
            break

        for tc in tool_calls:
            name     = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            log_t(f"-> Calling tool  : {BOLD}{name}{RESET}{TEAL}")
            log_gr(f"  Input snippet : {json.dumps(args)[:180]}")

            if name not in TOOL_REGISTRY:
                # Feed the mistake back so the agent can correct itself.
                result_content = json.dumps({
                    "status":  "RETRY",
                    "message": (
                        f"No such tool: {name}. Available tools: "
                        f"{', '.join(TOOL_REGISTRY.keys())}."
                    ),
                })
                log_a(f"Unknown tool: {name} — returning RETRY so the agent can correct.")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_content})
                div()
                continue

            result         = invoke_tool(TOOL_REGISTRY[name], args)
            result_content = result.model_dump_json()

            log_status(result.status.value)
            log_gr(f"  Message       : {result.message[:220]}")

            if result.status == AgentStatus.ESCALATE:
                log_r(f"\nESCALATE — '{name}' cannot proceed.")
                log_r(f"Reason: {result.message}")
                div()
                banner("Pipeline Halted — Schema Drift Detected")
                log_r("Migration report cannot be generated until eval schema is corrected.")
                log_gr("Airflow task marked failed. Audit trail preserved.")
                print()
                sys.exit(1)

            if result.status == AgentStatus.SUCCESS:
                called.append(name)

            # The demo beat: the report streams as soon as the agent drafts it.
            if name == "draft_migration_report":
                stream_report(result)

            if name == "generate_summary" and result.status == AgentStatus.SUCCESS:
                final_result = result

            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      trim_for_history(name, result_content),
            })
            div()

        if "generate_summary" in called:
            break

    # ── Fallback — only if the agent left required steps undone ────────────────
    missing = [t for t in ("draft_migration_report", "generate_summary") if t not in called]
    if missing:
        fallback_result = _deterministic_fallback(missing)
        final_result    = fallback_result or final_result

    if final_result is None:
        log_r("Agent loop ended without a final result — check output above.")
        sys.exit(1)

    banner("NemoClaw Eval Agent — Analysis Complete")
    log_g(f"Status  : {final_result.status.value}")
    log_g(f"Summary : {final_result.message[:400]}")
    if not missing:
        log_gr(f"All 6 steps executed by the agent — {len(called)} successful tool calls.")
    print()


def main():
    parser = argparse.ArgumentParser(description="AirClaw model eval demo runner")
    parser.add_argument("--break", dest="broken", action="store_true")
    args = parser.parse_args()

    if args.broken:
        shutil.copy(BROKEN, UPSTREAM)
        print(f"{AMBER}[setup] Swapped in broken eval data — 'model_b_quality_score' -> 'model_b_score'{RESET}")
        print(f"{AMBER}[setup] ESCALATE beat active{RESET}\n")
    else:
        shutil.copy(CLEAN, UPSTREAM)
        print(f"{TEAL}[setup] Clean eval data loaded — {UPSTREAM.name}{RESET}\n")

    api_key, key_error = get_nim_key()
    if key_error:
        print(f"{RED}Error: {key_error}{RESET}")
        sys.exit(1)

    run_agent(api_key)

if __name__ == "__main__":
    main()