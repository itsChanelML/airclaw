"""
NemoClawOperator — Airflow 3 compatible
---------------------------------------
Custom Airflow operator that runs the NemoClaw agent loop inside a task.

The agent reasons over a goal, calls typed tools from a registry, and returns
a structured result to XCom. ESCALATE fails the task with a typed diagnosis
rather than a stack trace.

Works with any tool registry exposing TOOL_REGISTRY, TOOL_SCHEMAS,
AgentStatus, AgentResult, and (optionally) trim_for_history:
  - airclaw_tools     — NYC 311 triage
  - model_eval_tools  — model migration eval

Airflow 3 notes:
  - apply_defaults was removed in Airflow 3; BaseOperator handles defaults.
  - Data and tool paths resolve through airclaw_env.repo_root() so the DAG
    works whether it runs from the repo or from a copy in AIRFLOW_HOME.

Python 3.9 compatible.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from airflow.models import BaseOperator

# Repo root is importable regardless of where Airflow loaded this plugin from.
_PLUGIN_DIR = Path(__file__).resolve().parent
for _candidate in (_PLUGIN_DIR.parent, *_PLUGIN_DIR.parents):
    if (_candidate / "airclaw_env.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from airclaw_env import get_model, get_nim_key, repo_root  # noqa: E402

# ── ANSI colors ────────────────────────────────────────────────────────────────
TEAL   = "\033[38;5;43m"
PURPLE = "\033[38;5;135m"
AMBER  = "\033[38;5;214m"
GREEN  = "\033[38;5;82m"
RED    = "\033[38;5;196m"
GRAY   = "\033[38;5;245m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Resolved from NIM_MODEL / airclaw_env so a model EOL is a one-line fix.
DEFAULT_MODEL = get_model()

# Tool results whose payload is worth streaming into the task log — this is
# what makes the Airflow UI a viable demo surface.
STREAMED_PAYLOADS = {
    "draft_supervisor_briefing": ("briefing", "BRIEFING CONTENT (ready to send)"),
    "draft_migration_report":    ("report",   "MIGRATION REPORT"),
}

# Per-registry system prompts. The operator used to hardcode the 311 prompt,
# which meant the eval DAG was told to call draft_supervisor_briefing — a tool
# it does not have.
SYSTEM_PROMPTS = {
    "airclaw_tools": (
        "You are NemoClaw — an autonomous operations agent for NYC 311 service "
        "requests. You triage overnight data, find SLA breaches, detect spikes, "
        "draft supervisor briefings, and surface only what requires human "
        "attention.\n\n"
        "Rules:\n"
        "1. Call validate_schema first. If it ESCALATEs, stop.\n"
        "2. Call check_sla_breaches to find overdue cases. Its result tells you "
        "which agencies have breaches and who each supervisor is — use those "
        "exact names, do not invent them.\n"
        "3. Call detect_complaint_spike to find overnight anomalies.\n"
        "4. query_requests answers counting questions about the feed (group by "
        "complaint_type, borough, district, agency, status, or supervisor, with "
        "filters such as only_breaches). Use it for anything the other tools do "
        "not directly answer.\n"
        "5. Call draft_supervisor_briefing once per agency that has breaches, "
        "passing that agency's real supervisor name and the same file_path. "
        "Start with the highest breach counts.\n"
        "6. Call generate_summary last, passing file_path and the supervisors "
        "you briefed.\n"
        "7. One tool call per turn. No prose between calls. If a tool returns "
        "RETRY, read the message, fix the arguments, and call it again."
    ),
    "model_eval_tools": (
        "You are NemoClaw — an autonomous model evaluation agent. You compare "
        "two models on production eval data and produce a go/no-go migration "
        "recommendation backed by evidence.\n\n"
        "Rules:\n"
        "1. Call validate_schema first. If it ESCALATEs, stop.\n"
        "2. Call score_comparison, then detect_regression, then cost_analysis. "
        "Each takes file_path.\n"
        "3. Call draft_migration_report with the two model names and file_path. "
        "It re-reads the eval file itself — do not echo earlier tool payloads "
        "back to it.\n"
        "4. Call generate_summary last.\n"
        "5. One tool call per turn. No prose between calls. If a tool returns "
        "RETRY, read the message, fix the arguments, and call it again."
    ),
}

GENERIC_SYSTEM_PROMPT = (
    "You are NemoClaw — an autonomous pipeline agent. Accomplish the goal by "
    "calling the tools available to you, one per turn, with no prose between "
    "calls. If a tool returns RETRY, read the message, fix your arguments, and "
    "call it again. Stop when the goal is met."
)


class NemoClawOperator(BaseOperator):
    """
    Airflow operator that delegates task execution to a NemoClaw agent.

    Parameters
    ----------
    goal : str
        Natural language description of what the agent must accomplish.
    context : dict
        Context payload passed to the agent (file paths, required fields, etc).
    tools_module : str
        Tool registry module to import — "airclaw_tools" or "model_eval_tools".
    system_prompt : str, optional
        Override the per-registry default system prompt.
    model : str
        NIM model id.
    nim_endpoint : str
        NIM API base URL.
    nim_api_key_env : str
        Env var holding the NIM API key. Loaded from .env if present.
    max_iterations : int
        Agent turn limit before the task gives up.
    """

    template_fields = ("goal", "context")
    ui_color        = "#1D9E75"

    def __init__(
        self,
        goal:            str,
        context:         Optional[Dict[str, Any]] = None,
        tools_module:    str = "airclaw_tools",
        system_prompt:   Optional[str] = None,
        model:           Optional[str] = None,
        nim_endpoint:    str = "https://integrate.api.nvidia.com/v1",
        nim_api_key_env: str = "NIM_API_KEY",
        max_iterations:  int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.goal            = goal
        self.context         = context if context is not None else {}
        self.tools_module    = tools_module
        self.system_prompt   = system_prompt
        self.model           = model or get_model()
        self.nim_endpoint    = nim_endpoint
        self.nim_api_key_env = nim_api_key_env
        self.max_iterations  = max_iterations

    # ── Main ───────────────────────────────────────────────────────────────────

    def execute(self, context: Any) -> dict:
        self._banner("NemoClaw Agent — Starting")
        self._log_teal(f"Goal       : {self.goal[:120]}…")
        self._log_teal(f"Model      : {self.model}")
        self._log_teal(f"Tools      : {self.tools_module}")
        self._log_divider()

        api_key, key_error = get_nim_key(self.nim_api_key_env)
        if key_error:
            raise ValueError(f"[AirClaw] {key_error}")

        mod = self._load_tools()
        tool_registry = mod.TOOL_REGISTRY
        tool_schemas  = mod.TOOL_SCHEMAS
        AgentStatus   = mod.AgentStatus
        trim          = getattr(mod, "trim_for_history", lambda name, payload: payload)

        self._log_gray(f"Tools loaded: {list(tool_registry.keys())}")
        self._log_divider()

        messages = [
            {"role": "system", "content": self._resolve_system_prompt()},
            {"role": "user",   "content": self._user_message()},
        ]

        final_result  = None
        called: List[str] = []

        for iteration in range(1, self.max_iterations + 1):
            self._log_purple(f"[Iteration {iteration}] Calling NemoClaw…")

            response = self._call_nim(api_key, messages, tool_schemas)
            if response is None:
                self._log_amber("NIM call failed — retrying in 2s…")
                time.sleep(2)
                continue

            assistant_msg = response["choices"][0]["message"]
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls", [])

            if not tool_calls:
                thought = assistant_msg.get("content", "")
                self._log_green("Agent completed reasoning — no further tool calls.")
                if thought:
                    self._log_gray(f"Final thought: {thought[:300]}")
                self._log_divider()
                break

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                raw_args  = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                self._log_teal(f"→ Calling tool  : {BOLD}{tool_name}{RESET}{TEAL}")
                self._log_gray( f"  Input snippet : {json.dumps(args)[:180]}")

                if tool_name not in tool_registry:
                    # Hand the mistake back as RETRY so the agent can recover
                    # instead of the task dying on a hallucinated tool name.
                    payload = json.dumps({
                        "status":  "RETRY",
                        "message": (
                            f"No such tool: {tool_name}. Available tools: "
                            f"{', '.join(tool_registry.keys())}."
                        ),
                    })
                    self._log_amber(f"Unknown tool: {tool_name} — returning RETRY.")
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"], "content": payload,
                    })
                    self._log_divider()
                    continue

                tool_result = self._invoke_tool(tool_registry[tool_name], args, mod)
                result_json = tool_result.model_dump_json()

                self._log_status(tool_result.status.value)
                self._log_gray(f"  Message       : {tool_result.message[:200]}")

                self._stream_payload(tool_name, tool_result)

                if tool_result.status == AgentStatus.ESCALATE:
                    self._log_red(f"\nESCALATE — {tool_name} could not complete.")
                    self._log_red(f"Reason: {tool_result.message}")
                    self._log_divider()
                    raise RuntimeError(
                        f"[AirClaw ESCALATE] {tool_name}: {tool_result.message}"
                    )

                if tool_result.status == AgentStatus.SUCCESS:
                    called.append(tool_name)
                    final_result = tool_result

                # Trim before adding to history — untrimmed payloads blow the
                # context window and make the agent stall mid-pipeline.
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      trim(tool_name, result_json),
                })
                self._log_divider()

            if "generate_summary" in called:
                break

        if final_result is None:
            raise RuntimeError(
                "[AirClaw] Agent loop ended without a successful tool result."
            )

        self._banner("NemoClaw Agent — Complete")
        self._log_green(f"Status  : {final_result.status.value}")
        self._log_green(f"Summary : {final_result.message[:400]}")
        self._log_gray(f"Tool calls executed: {' → '.join(called)}")
        self._log_divider()

        return final_result.model_dump()

    # ── Tool registry loading ──────────────────────────────────────────────────

    def _load_tools(self):
        tools_path = str(repo_root() / "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        try:
            return importlib.import_module(self.tools_module)
        except ImportError as e:
            raise ImportError(
                f"[AirClaw] Could not import tools module '{self.tools_module}' "
                f"from {tools_path}. Available: "
                f"{sorted(p.stem for p in Path(tools_path).glob('*_tools.py'))}. "
                f"Original error: {e}"
            )

    # ── NIM call ───────────────────────────────────────────────────────────────

    def _call_nim(self, api_key, messages, tools):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "messages":    messages,
            "tools":       tools,
            "tool_choice": "auto",
            "max_tokens":  1024,
            "temperature": 0.1,
        }
        # Escalating timeouts — the 49B model can be slow to first token under
        # load, and a single 90s timeout was enough to kill a task mid-run.
        for attempt, timeout in enumerate((120, 150, 180), 1):
            try:
                resp = requests.post(
                    f"{self.nim_endpoint}/chat/completions",
                    headers=headers, json=payload, timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < 3:
                    self._log_amber(
                        f"NIM timeout (attempt {attempt}/3) — retrying with a "
                        f"longer timeout…"
                    )
                    time.sleep(3)
                else:
                    self._log_red(f"NIM error: timed out after {timeout}s")
                    return None
            except requests.exceptions.RequestException as e:
                self._log_red(f"NIM API error: {e}")
                return None

    # ── Tool invocation ────────────────────────────────────────────────────────

    def _invoke_tool(self, tool_fn, args, mod):
        import inspect
        params = list(inspect.signature(tool_fn).parameters.values())
        if params:
            input_model = params[0].annotation
            try:
                return tool_fn(input_model(**args))
            except Exception as e:
                # RETRY, not failure — the agent gets the validation error back
                # and can correct its arguments.
                return mod.AgentResult(
                    status=mod.AgentStatus.RETRY,
                    message=f"Tool input error: {e}",
                    tool=getattr(tool_fn, "__name__", "unknown"),
                )
        return tool_fn(args)

    # ── Prompts ────────────────────────────────────────────────────────────────

    def _resolve_system_prompt(self) -> str:
        if self.system_prompt:
            return self.system_prompt
        return SYSTEM_PROMPTS.get(self.tools_module, GENERIC_SYSTEM_PROMPT)

    def _user_message(self) -> str:
        return (
            f"Goal: {self.goal}\n\n"
            f"Context:\n{json.dumps(self.context, indent=2, default=str)}\n\n"
            "Begin. Start with schema validation."
        )

    # ── Logging ────────────────────────────────────────────────────────────────

    def _stream_payload(self, tool_name, tool_result):
        spec = STREAMED_PAYLOADS.get(tool_name)
        if not spec:
            return
        key, label = spec
        body = tool_result.data.get(key)
        if not body:
            return
        self._log_divider()
        self._log_teal(f"  {label}:")
        for line in body.split("\n"):
            self._log_gray(f"  {line}")
        self._log_divider()

    def _banner(self, text):
        w = 62
        self.log.info(f"{BOLD}{TEAL}{'─'*w}{RESET}")
        self.log.info(f"{BOLD}{TEAL}  {text}{RESET}")
        self.log.info(f"{BOLD}{TEAL}{'─'*w}{RESET}")

    def _log_divider(self): self.log.info(f"{GRAY}{'·'*52}{RESET}")
    def _log_teal(self,   m): self.log.info(f"{TEAL}{m}{RESET}")
    def _log_purple(self, m): self.log.info(f"{PURPLE}{m}{RESET}")
    def _log_green(self,  m): self.log.info(f"{GREEN}{m}{RESET}")
    def _log_amber(self,  m): self.log.info(f"{AMBER}{m}{RESET}")
    def _log_red(self,    m): self.log.info(f"{RED}{BOLD}{m}{RESET}")
    def _log_gray(self,   m): self.log.info(f"{GRAY}{m}{RESET}")

    def _log_status(self, s):
        c = GREEN if s == "SUCCESS" else (AMBER if s == "RETRY" else RED)
        self.log.info(f"{c}{BOLD}  Status        : {s}{RESET}")
