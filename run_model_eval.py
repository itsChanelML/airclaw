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
from model_eval_tools import TOOL_REGISTRY, TOOL_SCHEMAS, AgentStatus
import requests

TEAL   = "\033[38;5;43m"
PURPLE = "\033[38;5;135m"
AMBER  = "\033[38;5;214m"
GREEN  = "\033[38;5;82m"
RED    = "\033[38;5;196m"
GRAY   = "\033[38;5;245m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

NIM_MODEL    = "nvidia/llama-3.3-nemotron-super-49b-v1"
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

def trim_for_history(tool_name, result_json):
    try:
        obj  = json.loads(result_json)
        data = obj.get("data", {})
        if tool_name == "score_comparison" and data:
            by_cat = {}
            for cat, d in data.get("by_category", {}).items():
                by_cat[cat] = {
                    "model_a_quality": d.get("model_a_quality"),
                    "model_b_quality": d.get("model_b_quality"),
                    "quality_delta":   d.get("quality_delta"),
                    "winner":          d.get("winner"),
                }
            obj["data"] = {
                "by_category":     by_cat,
                "overall_delta":   data.get("overall_delta"),
                "model_b_wins":    data.get("model_b_wins", []),
                "model_a_wins":    data.get("model_a_wins", []),
                "model_a_name":    data.get("model_a_name", ""),
                "model_b_name":    data.get("model_b_name", ""),
                "total_evaluated": data.get("total_evaluated", 0),
            }
        elif tool_name == "detect_regression" and data:
            obj["data"] = {
                "regressions":      data.get("regressions", [])[:3],
                "refusal_spikes":   data.get("refusal_spikes", [])[:3],
                "hold_categories":  data.get("hold_categories", []),
                "clear_to_migrate": data.get("clear_to_migrate", False),
            }
        else:
            obj.pop("data", None)
        return json.dumps(obj)
    except Exception:
        return result_json[:600]

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

def _run_final_steps(messages, cost_data):
    """
    Agent stops after cost_analysis — invoke final 2 steps directly.
    cost_data is the FULL result.data (not trimmed) so $savings displays correctly.
    """
    from model_eval_tools import (
        draft_migration_report, generate_summary,
        DraftMigrationReportInput, GenerateSummaryInput
    )

    scores_data      = {}
    regressions_data = {}

    for msg in messages:
        if msg.get("role") == "tool":
            try:
                obj  = json.loads(msg["content"])
                data = obj.get("data", {})
                if "by_category" in data:
                    scores_data = data
                elif "hold_categories" in data:
                    regressions_data = data
            except Exception:
                pass

    # Step 5
    log_p("[Iteration 5] Calling NemoClaw…")
    log_t(f"-> Calling tool  : {BOLD}draft_migration_report{RESET}{TEAL}")
    r = draft_migration_report(DraftMigrationReportInput(
        model_a="gpt-4o",
        model_b="nvidia/llama-3.3-nemotron-super-49b-v1",
        scores=scores_data,
        regressions=regressions_data,
        cost_analysis=cost_data,
        evaluated_on=300,
        as_of=datetime.now().strftime("%B %d, %Y"),
    ))
    log_status(r.status.value)
    log_gr(f"  Message       : {r.message[:220]}")

    if r.data.get("report"):
        div()
        log_t("  MIGRATION REPORT:")
        print()
        for line in r.data["report"].split("\n"):
            print(f"  {GRAY}{line}{RESET}")
        print()
        div()

    # Step 6
    log_p("[Iteration 6] Calling NemoClaw…")
    log_t(f"-> Calling tool  : {BOLD}generate_summary{RESET}{TEAL}")
    s = generate_summary(GenerateSummaryInput(
        recommendation=r.data.get("recommendation", ""),
        model_a="gpt-4o",
        model_b="nvidia/llama-3.3-nemotron-super-49b-v1",
        total_evaluated=300,
    ))
    log_status(s.status.value)
    log_gr(f"  Message       : {s.message[:220]}")
    div()

    banner("NemoClaw Eval Agent — Analysis Complete")
    log_g(f"Status  : {s.status.value}")
    log_g(f"Summary : {s.message[:400]}")
    print()

def run_agent(api_key):
    banner("NemoClaw Eval Agent — Model Migration Analysis Starting")
    log_t(f"Comparing : gpt-4o  ->  {NIM_MODEL}")
    log_t(f"Prompts   : 300 production samples across 4 task categories")
    log_t(f"As of     : {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    div()

    fp         = str(UPSTREAM.resolve())
    req_fields = json.dumps(REQUIRED_FIELDS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Analyze model migration from gpt-4o to nvidia/llama-3.3-nemotron-super-49b-v1.\n\n"
            f"File: {fp}\n\n"
            f"Call validate_schema with these EXACT required_fields — do not change them:\n{req_fields}\n\n"
            "Then call these tools in order:\n"
            "2. score_comparison (file_path only)\n"
            "3. detect_regression (file_path only)\n"
            "4. cost_analysis (file_path, monthly_volume=100000)\n\n"
            "Start now. Call validate_schema."
        )},
    ]

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
            log_g("Agent completed reasoning.")
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

            if name not in TOOL_REGISTRY:
                result_content = json.dumps({"status": "ESCALATE", "message": f"Unknown tool: {name}"})
                log_r(f"Unknown tool: {name}")
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

            # After cost_analysis: run final steps directly with FULL data
            if name == "cost_analysis" and result.status == AgentStatus.SUCCESS:
                div()
                _run_final_steps(messages, result.data)
                return

            history_content = trim_for_history(name, result_content)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      history_content,
            })
            div()

    log_r("Agent loop ended — check output above.")
    sys.exit(1)

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

    api_key = os.environ.get("NIM_API_KEY")
    if not api_key:
        print(f"{RED}Error: NIM_API_KEY not set.{RESET}")
        sys.exit(1)

    run_agent(api_key)

if __name__ == "__main__":
    main()