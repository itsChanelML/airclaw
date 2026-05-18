#!/usr/bin/env python3
"""
AirClaw Model Eval Runner — GPT-4o vs Nemotron-Super
-----------------------------------------------------
Runs the full model migration eval agent in your terminal.
Use this to record your backup demo.

Usage:
  python run_model_eval.py               # happy path — full report
  python run_model_eval.py --break       # schema drift → ESCALATE beat

Environment:
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
MAX_ITER     = 14   # eval pipeline has 6 tool calls — give headroom

REQUIRED_FIELDS = [
    "prompt_id", "category", "model_a", "model_b",
    "model_a_quality_score", "model_b_quality_score",
    "model_a_format_score", "model_b_format_score",
    "model_a_latency_sec", "model_b_latency_sec",
    "model_a_cost_usd", "model_b_cost_usd",
    "model_a_refused", "model_b_refused", "regression_flag",
]

GOAL = (
    "You are a model migration eval agent. A team is considering migrating "
    "from GPT-4o to nvidia/llama-3.3-nemotron-super-49b-v1 in production. "
    "Your job is to do the analysis that would take a Sr Engineer 2-3 days manually "
    "and produce a go/no-go recommendation with evidence. "
    "Step 1: Validate the eval file schema — stop immediately if it has drifted. "
    "Step 2: Run score_comparison to compare model quality by task category. "
    "Step 3: Run detect_regression to find where Model B meaningfully regresses. "
    "Step 4: Run cost_analysis at 100,000 prompts/month to project savings. "
    "Step 5: Run draft_migration_report with the full data from all prior tool calls. "
    "Step 6: Run generate_summary. "
    "Be precise. Surface evidence. The VP of Engineering reads this in 5 minutes."
)

SYSTEM_PROMPT = (
    "You are NemoClaw — an autonomous model evaluation agent. "
    "You MUST call tools for every step. Never write analysis or reports as text.\n\n"
    "MANDATORY SEQUENCE — execute every step, no exceptions:\n"
    "STEP 1: Call validate_schema. If result is ESCALATE, stop.\n"
    "STEP 2: Call score_comparison. Store the full result data.\n"
    "STEP 3: Call detect_regression. Store the full result data.\n"
    "STEP 4: Call cost_analysis with monthly_volume=100000. Store the full result data.\n"
    "STEP 5: Call draft_migration_report. "
    "Pass scores=<step2 data>, regressions=<step3 data>, cost_analysis=<step4 data>, "
    "evaluated_on=<row_count from step2>, model_a='gpt-4o', "
    "model_b='nvidia/llama-3.3-nemotron-super-49b-v1'. "
    "DO NOT write the report as text. Call the tool.\n"
    "STEP 6: Call generate_summary with the recommendation from step 5.\n\n"
    "You are INCOMPLETE until generate_summary is called. "
    "Do not stop early. Do not summarize in text. Call every tool."
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
def log_r(m):   print(f"{RED}{BOLD}{m}{RESET}")
def log_gr(m):  print(f"{GRAY}{m}{RESET}")

def log_status(s):
    c = GREEN if s=="SUCCESS" else (AMBER if s=="RETRY" else RED)
    print(f"{c}{BOLD}  Status        : {s}{RESET}")

def call_nim(api_key, messages):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       NIM_MODEL,
        "messages":    messages,
        "tools":       TOOL_SCHEMAS,
        "tool_choice": "auto",
        "max_tokens":  2048,
        "temperature": 0.1,
    }
    try:
        r = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
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

def run_agent(api_key):
    banner("NemoClaw Eval Agent — Model Migration Analysis Starting")
    log_t(f"Comparing : gpt-4o  →  {NIM_MODEL}")
    log_t(f"Prompts   : 300 production samples across 4 task categories")
    log_t(f"As of     : {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    div()

    context = {
        "file_path":       str(UPSTREAM.resolve()),
        "required_fields": REQUIRED_FIELDS,
        "monthly_volume":  100000,
        "as_of":           datetime.now().strftime("%B %d, %Y"),
        "description":     "Production model migration eval — GPT-4o vs Nemotron-Super-49B",
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Goal: {GOAL}\n\nContext:\n{json.dumps(context, indent=2)}\n\nBegin analysis."},
    ]

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

            log_t(f"→ Calling tool  : {BOLD}{name}{RESET}{TEAL}")

            if name not in TOOL_REGISTRY:
                result_content = json.dumps({"status":"ESCALATE","message":f"Unknown tool: {name}"})
                log_r(f"Unknown tool: {name}")
            else:
                result = invoke_tool(TOOL_REGISTRY[name], args)
                result_content = result.model_dump_json()

                log_status(result.status.value)
                log_gr(f"  Message       : {result.message[:220]}")

                # Stream the migration report through logs — the demo wow moment
                if name == "draft_migration_report" and result.data.get("report"):
                    div()
                    log_t("  MIGRATION REPORT:")
                    print()
                    for line in result.data["report"].split("\n"):
                        print(f"  {GRAY}{line}{RESET}")
                    print()
                    div()

                if result.status == AgentStatus.ESCALATE:
                    log_r(f"\nESCALATE — '{name}' cannot proceed.")
                    log_r(f"Reason: {result.message}")
                    div()
                    banner("Pipeline Halted — Schema Drift Detected")
                    log_r("Migration report cannot be generated until eval schema is corrected.")
                    log_gr("Airflow task marked failed. Audit trail preserved.")
                    print()
                    sys.exit(1)

                final_result = result

            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result_content,
            })
            div()

    if final_result:
        banner("NemoClaw Eval Agent — Analysis Complete")
        log_g(f"Status  : {final_result.status.value}")
        log_g(f"Summary : {final_result.message[:400]}")
        print()
    else:
        log_r("Agent loop ended without a final result.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AirClaw model eval demo runner")
    parser.add_argument("--break", dest="broken", action="store_true",
                        help="Swap in broken eval schema → ESCALATE beat")
    args = parser.parse_args()

    if args.broken:
        shutil.copy(BROKEN, UPSTREAM)
        print(f"{AMBER}[setup] Swapped in broken eval data — 'model_b_quality_score' → 'model_b_score'{RESET}")
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