#!/usr/bin/env python3
"""
AirClaw Standalone Runner — NYC 311 Productivity Agent
-------------------------------------------------------
Runs the full NemoClaw triage loop directly in your terminal.
No Airflow needed. Use this for:
  - Testing before the conference
  - Backup demo if Airflow UI has issues
  - Audience input mode (pass a custom goal)

Usage:
  python3 run_demo.py                          # happy path
  python3 run_demo.py --break                  # schema drift → ESCALATE beat
  python3 run_demo.py --goal "your goal here"  # audience input mode

Environment:
  export NIM_API_KEY=your_key
  python3 run_demo.py
"""

import argparse, inspect, json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
TOOLS_DIR = ROOT / "tools"
UPSTREAM  = DATA_DIR / "nyc_311_upstream.csv"
CLEAN     = DATA_DIR / "nyc_311_clean.csv"
BROKEN    = DATA_DIR / "nyc_311_broken.csv"

sys.path.insert(0, str(TOOLS_DIR))
import airclaw_tools as tools_mod
from airclaw_tools import TOOL_REGISTRY, TOOL_SCHEMAS, AgentStatus

import requests

# ── ANSI ───────────────────────────────────────────────────────────────────────
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
MAX_ITER     = 14

REQUIRED_FIELDS = [
    "unique_key", "created_date", "complaint_type", "borough", "district",
    "agency", "supervisor", "status", "closed_date", "hours_open",
    "sla_hours", "sla_breach",
]

DEFAULT_GOAL = (
    "Run the NYC 311 morning triage pipeline. "
    "Call tools in this exact order: validate_schema, check_sla_breaches, "
    "detect_complaint_spike, then prioritize_queue and draft_supervisor_briefing "
    "for the TOP 2 agencies by breach count only, then generate_summary. "
    "Do not plan. Do not write text. Call the next tool immediately after each result."
)

# ── Helpers ────────────────────────────────────────────────────────────────────

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

# ── Tool result trimmer ────────────────────────────────────────────────────────
# Strips the nested data payload from tool results before adding to message
# history. The agent only needs status + message to reason forward.
# The tools read directly from the CSV file when they need actual case data.

def trim_result_for_history(tool_name: str, result_json: str) -> str:
    """
    Keep status + message always.
    For check_sla_breaches: also keep by_agency summary (agency name,
    count, supervisor, cases list) so the agent knows what to pass to
    prioritize_queue. Strip only the top-level breaches list.
    For all other tools: strip data entirely.
    """
    try:
        obj = json.loads(result_json)
        data = obj.get("data", {})

        if tool_name == "check_sla_breaches" and data:
            # Keep by_agency but trim each agency cases to first 3
            by_agency = data.get("by_agency", {})
            trimmed_by_agency = {}
            for agency, info in by_agency.items():
                trimmed_by_agency[agency] = {
                    "count":      info.get("count", 0),
                    "supervisor": info.get("supervisor", ""),
                    "cases":      info.get("cases", [])[:3],  # first 3 cases only
                }
            obj["data"] = {
                "by_agency": trimmed_by_agency,
                "total":     data.get("total", 0),
                "agencies":  data.get("agencies", []),
            }
        else:
            obj.pop("data", None)

        return json.dumps(obj)
    except Exception:
        return result_json[:800]

# ── NIM API ────────────────────────────────────────────────────────────────────

def call_nim(api_key: str, messages: list, tool_schemas: list):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       NIM_MODEL,
        "messages":    messages,        # full history — do NOT trim here
        "tools":       tool_schemas,
        "tool_choice": "auto",
        "max_tokens":  1024,
        "temperature": 0.1,
    }
    try:
        r = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log_r(f"NIM error: {e}")
        return None

# ── Tool invocation ────────────────────────────────────────────────────────────

def invoke_tool(fn, args: dict):
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

# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(goal: str, context: dict, api_key: str):
    banner("NemoClaw Agent — Morning Triage Starting")
    log_t(f"Goal       : {goal[:110]}…")
    log_t(f"Model      : {NIM_MODEL}")
    log_t(f"As of      : {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    div()

    system_prompt = (
        "You are NemoClaw. Call tools one at a time. Never write analysis or output as text."
        "Rule 1: After every tool result, immediately call the next tool. No planning text."
        "Rule 2: validate_schema → check_sla_breaches → detect_complaint_spike"
        "Rule 3: Then call prioritize_queue for the top 2 agencies by breach count."
        "Rule 4: Then call draft_supervisor_briefing for those same 2 agencies."
        "Rule 5: Then call generate_summary. You are done only when generate_summary returns."
        "Rule 6: If you are about to write text instead of calling a tool, stop and call the tool."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"Goal: {goal}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            "Begin triage. Start with validate_schema."
        )},
    ]

    final_result = None

    for iteration in range(1, MAX_ITER + 1):
        log_p(f"[Iteration {iteration}] Calling NemoClaw…")

        response = call_nim(api_key, messages, TOOL_SCHEMAS)
        if response is None:
            log_a("NIM call failed — retrying in 2s…")
            time.sleep(2)
            continue

        assistant_msg = response["choices"][0]["message"]
        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            thought = assistant_msg.get("content", "")
            log_g("Agent completed reasoning — no further tool calls.")
            if thought:
                log_gr(f"Final thought: {thought[:300]}")
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
            log_gr(f"  Input snippet : {json.dumps(args)[:180]}")

            if name not in TOOL_REGISTRY:
                result_content  = json.dumps({"status": "ESCALATE", "message": f"Unknown tool: {name}"})
                history_content = result_content
                log_r(f"Unknown tool: {name}")
            else:
                result         = invoke_tool(TOOL_REGISTRY[name], args)
                result_content = result.model_dump_json()

                # What goes into history = status + message only (no nested data)
                # What gets displayed   = full result including briefing text
                history_content = trim_result_for_history(name, result_content)

                log_status(result.status.value)
                log_gr(f"  Message       : {result.message[:220]}")

                # Stream briefing to terminal — the demo wow moment
                if name == "draft_supervisor_briefing" and result.data.get("briefing"):
                    div()
                    log_t("  BRIEFING READY TO SEND:")
                    print()
                    for line in result.data["briefing"].split("\n"):
                        print(f"  {GRAY}{line}{RESET}")
                    print()
                    div()

                if result.status == AgentStatus.ESCALATE:
                    log_r(f"\nESCALATE — '{name}' cannot proceed.")
                    log_r(f"Reason: {result.message}")
                    div()
                    banner("Pipeline Halted — Escalating to Orchestrator")
                    log_r("Task flagged in Airflow. Audit trail preserved.")
                    log_gr("Supervisor notified. No silent failure.")
                    print()
                    sys.exit(1)

                final_result = result

            # Use trimmed content in history to keep context window lean
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      history_content,
            })
            div()

    # ── Final output ───────────────────────────────────────────────────────────
    if final_result:
        banner("NemoClaw Agent — Triage Complete")
        log_g(f"Status  : {final_result.status.value}")
        log_g(f"Summary : {final_result.message[:500]}")
        if final_result.data:
            print()
            log_t("Structured XCom output:")
            print(json.dumps(final_result.data, indent=2)[:800])
        print()
    else:
        log_r("Agent loop ended without a final result.")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AirClaw NYC 311 productivity agent demo")
    parser.add_argument("--break", dest="broken", action="store_true",
                        help="Swap in broken schema → ESCALATE demo beat")
    parser.add_argument("--goal",  type=str, default=DEFAULT_GOAL,
                        help="Custom goal (audience input mode)")
    args = parser.parse_args()

    if args.broken:
        shutil.copy(BROKEN, UPSTREAM)
        print(f"{AMBER}[setup] Swapped in broken data — 'complaint_type' → 'complaint_category'{RESET}")
        print(f"{AMBER}[setup] ESCALATE beat active{RESET}\n")
    else:
        shutil.copy(CLEAN, UPSTREAM)
        print(f"{TEAL}[setup] Clean data loaded — {UPSTREAM.name}{RESET}\n")

    api_key = os.environ.get("NIM_API_KEY")
    if not api_key:
        print(f"{RED}Error: NIM_API_KEY not set.{RESET}")
        print(f"{GRAY}Get your key: https://build.nvidia.com{RESET}")
        sys.exit(1)

    context = {
        "file_path":       str(UPSTREAM.resolve()),
        "required_fields": REQUIRED_FIELDS,
        "as_of":           datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "description":     "NYC 311 overnight triage — SLA monitoring and supervisor briefing pipeline",
        "source":          "data.cityofnewyork.us/resource/erm2-nwe9",
    }

    run_agent(goal=args.goal, context=context, api_key=api_key)


if __name__ == "__main__":
    main()