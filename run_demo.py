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
from airclaw_tools import (
    TOOL_REGISTRY, TOOL_SCHEMAS, AgentStatus,
    trim_for_history as trim_result_for_history,
)

sys.path.insert(0, str(ROOT))
from airclaw_env import get_model, get_nim_key
from rebase_data import describe_shift, rebase_csv

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

NIM_MODEL    = get_model()
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MAX_ITER     = 14

REQUIRED_FIELDS = [
    "unique_key", "created_date", "complaint_type", "borough", "district",
    "agency", "supervisor", "status", "closed_date", "hours_open",
    "sla_hours", "sla_breach",
]

DEFAULT_GOAL = (
    "It is 6am. Run the NYC 311 morning triage: find every open request that has "
    "breached its SLA window, check whether any complaint type spiked overnight, "
    "and draft a ready-to-send briefing for the supervisor of each agency carrying "
    "breaches — highest breach counts first, at most three agencies. "
    "Finish with a one-paragraph duty manager summary."
)

# Tool selection and ordering are the agent's job — the goal above says what to
# accomplish, not which functions to call. That is what makes --goal meaningful.
SYSTEM_PROMPT = (
    "You are NemoClaw, an autonomous operations agent for NYC 311.\n\n"
    "How you work:\n"
    "- Call validate_schema before anything else. If it ESCALATEs, stop.\n"
    "- check_sla_breaches tells you which agencies have overdue cases and names "
    "each agency's supervisor. Use those exact agency codes and supervisor names; "
    "never invent one.\n"
    "- draft_supervisor_briefing writes one briefing for one agency. Call it once "
    "per agency you decide needs one, passing agency, supervisor, and the same "
    "file_path used earlier. It reads the case data from the file itself.\n"
    "- query_requests answers counting questions about the data — group requests "
    "by complaint_type, borough, district, agency, status, or supervisor, with "
    "optional filters like only_breaches. Use it for anything the other tools do "
    "not directly answer, and base your answer on what it returns rather than "
    "guessing.\n"
    "- generate_summary is always your last call. Pass file_path, the list of "
    "supervisors you briefed, and overnight_total. The run is NOT complete until "
    "you call it — do not stop after the last briefing.\n\n"
    "Rules: one tool call per turn. No prose between calls. If a tool returns "
    "RETRY, read its message, fix your arguments, and call it again. Only call "
    "tools that exist in your tool list.\n\n"
    "If the goal is a question rather than the standard triage run, gather the "
    "facts with query_requests and then reply with a short, direct answer — two "
    "or three sentences citing the numbers. Do not narrate your reasoning."
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

# ── NIM API ────────────────────────────────────────────────────────────────────

def call_nim(api_key: str, messages: list, tool_schemas: list):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       NIM_MODEL,
        "messages":    messages,
        "tools":       tool_schemas,
        "tool_choice": "auto",
        "max_tokens":  1024,
        "temperature": 0.1,
    }
    # Retry up to 3 times with increasing timeout for slow NIM responses
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

def _fallback(missing, file_path: str, briefed=None, top_n: int = 2):
    """
    Safety net, not the happy path. The agent is expected to make these calls
    itself; this only runs for the steps it left undone, and every line is
    labeled so the logs never credit the agent with work it did not do.
    """
    from airclaw_tools import (
        check_sla_breaches, draft_supervisor_briefing, generate_summary,
        SLABreachInput, DraftBriefingInput, GenerateSummaryInput,
    )

    log_a(f"[fallback] Agent did not call: {', '.join(missing)}")
    log_a("[fallback] Completing those steps deterministically — not agent output.")
    div()

    breaches  = check_sla_breaches(SLABreachInput(file_path=file_path))
    by_agency = breaches.data.get("by_agency", {}) if breaches.data else {}
    total     = breaches.data.get("total", 0) if breaches.data else 0
    drafted   = list(briefed or [])

    if "draft_supervisor_briefing" in missing:
        ranked = sorted(by_agency.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)
        for agency, info in ranked[:top_n]:
            log_a(f"[fallback] Running tool : {BOLD}draft_supervisor_briefing{RESET}{AMBER} ({agency})")
            r = draft_supervisor_briefing(DraftBriefingInput(
                agency=agency,
                supervisor=info.get("supervisor", ""),
                file_path=file_path,
            ))
            log_status(r.status.value)
            if r.data.get("briefing"):
                div()
                log_t("  BRIEFING READY TO SEND:")
                print()
                for line in r.data["briefing"].split("\n"):
                    print(f"  {GRAY}{line}{RESET}")
                print()
                div()
                drafted.append(info.get("supervisor", agency))

    if "generate_summary" in missing:
        log_a(f"[fallback] Running tool : {BOLD}generate_summary{RESET}{AMBER}")
        summary = generate_summary(GenerateSummaryInput(
            file_path=file_path,
            briefings=drafted,
            overnight_total=total,
        ))
        log_status(summary.status.value)
        log_gr(f"  Message       : {summary.message[:220]}")
        div()
        return summary

    return None


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(goal: str, context: dict, api_key: str, custom_goal: bool = False):
    banner("NemoClaw Agent — Morning Triage Starting")
    log_t(f"Goal       : {goal[:110]}…")
    log_t(f"Model      : {NIM_MODEL}")
    log_t(f"As of      : {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    div()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Goal: {goal}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            "Begin triage. Start with validate_schema."
        )},
    ]

    final_result = None
    called       = []
    briefed      = []
    answer       = ""

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
            answer = (assistant_msg.get("content") or "").strip()
            log_g("Agent completed reasoning — no further tool calls.")
            if answer:
                # In audience-input mode the prose IS the deliverable, so show
                # all of it rather than a truncated grey aside.
                div()
                log_t("  AGENT ANSWER:")
                print()
                for line in answer.split("\n"):
                    print(f"  {GRAY}{line}{RESET}")
                print()
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
                # RETRY, not ESCALATE — a hallucinated tool name is recoverable.
                result_content  = json.dumps({
                    "status":  "RETRY",
                    "message": (
                        f"No such tool: {name}. Available tools: "
                        f"{', '.join(TOOL_REGISTRY.keys())}."
                    ),
                })
                history_content = result_content
                log_a(f"Unknown tool: {name} — returning RETRY so the agent can correct.")
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

                if result.status == AgentStatus.SUCCESS:
                    called.append(name)
                    if name == "draft_supervisor_briefing":
                        briefed.append(result.data.get("supervisor", ""))
                final_result = result

            # Use trimmed content in history to keep context window lean
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      history_content,
            })
            div()

        if "generate_summary" in called:
            break

    # ── Fallback ──────────────────────────────────────────────────────────────
    # The briefings are the demo, and generate_summary is the final XCom payload
    # the downstream Airflow task reads. If the agent skipped either, finish the
    # run deterministically rather than dying on stage.
    missing = [t for t in ("draft_supervisor_briefing", "generate_summary")
               if t not in called]

    # A custom goal that the agent answered in prose is complete as it stands —
    # forcing briefings nobody asked for would be noise, not a safety net. The
    # fallback exists for the standard pipeline, where the briefings ARE the job.
    if missing and custom_goal and answer:
        log_gr(f"Custom goal answered directly; skipped: {', '.join(missing)}.")
        missing = []

    if missing:
        final_result = _fallback(missing, context["file_path"], briefed) or final_result

    # ── Final output ───────────────────────────────────────────────────────────
    # A question-shaped goal ends on the answer, not a triage summary.
    if custom_goal and answer and "draft_supervisor_briefing" not in called:
        banner("NemoClaw Agent — Question Answered")
        log_g("The agent gathered the facts with its tools and answered directly.")
        log_gr(f"Tools used: {' → '.join(called)}")
        print()
        return

    if final_result:
        banner("NemoClaw Agent — Triage Complete")
        if not missing:
            log_gr(f"All steps executed by the agent: {' → '.join(called)}")
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

    # Rebase rather than copy: the sample's timestamps are frozen in the past,
    # and the SLA/spike windows are all relative to now. See rebase_data.py.
    source = BROKEN if args.broken else CLEAN
    shift  = rebase_csv(source, UPSTREAM)

    if args.broken:
        print(f"{AMBER}[setup] Swapped in broken data — 'complaint_type' → 'complaint_category'{RESET}")
        print(f"{AMBER}[setup] ESCALATE beat active{RESET}\n")
    else:
        print(f"{TEAL}[setup] Clean data loaded — {UPSTREAM.name}{RESET}")
        print(f"{GRAY}[setup] Dates rebased {describe_shift(shift)} — newest request is now{RESET}\n")

    api_key, key_error = get_nim_key()
    if key_error:
        print(f"{RED}Error: {key_error}{RESET}")
        sys.exit(1)

    context = {
        "file_path":       str(UPSTREAM.resolve()),
        "required_fields": REQUIRED_FIELDS,
        "as_of":           datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "description":     "NYC 311 overnight triage — SLA monitoring and supervisor briefing pipeline",
        "source":          "data.cityofnewyork.us/resource/erm2-nwe9",
    }

    run_agent(
        goal=args.goal,
        context=context,
        api_key=api_key,
        custom_goal=(args.goal != DEFAULT_GOAL),
    )


if __name__ == "__main__":
    main()