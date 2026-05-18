"""
NemoClawOperator (updated for productivity agent)
-------------------------------------------------
Custom Airflow operator that runs the NemoClaw agent loop.
Updated to support a tools_module parameter so it can load
new_airclaw_tools (or any future tool set) dynamically.

Key additions vs original:
  - tools_module param: load any tools file by name
  - Plain-English logs tuned to productivity agent context
  - Briefing content streams through logs in real time
"""

import importlib
import json
import os
import sys
import time
from typing import Any

import requests
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

# ── ANSI colors ────────────────────────────────────────────────────────────────
TEAL   = "\033[38;5;43m"
PURPLE = "\033[38;5;135m"
AMBER  = "\033[38;5;214m"
GREEN  = "\033[38;5;82m"
RED    = "\033[38;5;196m"
GRAY   = "\033[38;5;245m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

NIM_MODEL  = "nvidia/nemotron-super-49b-v1"
MAX_ITER   = 10


class NemoClawOperator(BaseOperator):
    """
    Airflow operator that delegates task execution to a NemoClaw agent.

    Parameters
    ----------
    goal : str
        Natural language description of what the agent must accomplish.
    context : dict
        Context payload passed to the agent.
    tools_module : str
        Name of the tools module to import (default: "new_airclaw_tools").
    nim_endpoint : str
        NIM API base URL.
    nim_api_key_env : str
        Env var name holding the NIM API key.
    max_retries : int
        Agent-level retry limit before escalating.
    """

    template_fields = ("goal", "context")
    ui_color        = "#1D9E75"

    @apply_defaults
    def __init__(
        self,
        goal:            str,
        context:         dict[str, Any] = {},
        tools_module:    str  = "new_airclaw_tools",
        nim_endpoint:    str  = "https://integrate.api.nvidia.com/v1",
        nim_api_key_env: str  = "NIM_API_KEY",
        max_retries:     int  = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.goal            = goal
        self.context         = context
        self.tools_module    = tools_module
        self.nim_endpoint    = nim_endpoint
        self.nim_api_key_env = nim_api_key_env
        self.max_retries     = max_retries

    # ── Main ───────────────────────────────────────────────────────────────────

    def execute(self, context: Any) -> dict:
        self._banner("NemoClaw Agent — Morning Triage Starting")
        self._log_teal(f"Goal       : {self.goal[:120]}…")
        self._log_teal(f"Model      : {NIM_MODEL}")
        self._log_divider()

        api_key = os.environ.get(self.nim_api_key_env)
        if not api_key:
            raise ValueError(
                f"NIM API key not found. Set: export {self.nim_api_key_env}=your_key"
            )

        # Load tool registry dynamically
        tools_path = os.path.join(os.path.dirname(__file__), "..", "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        mod           = importlib.import_module(self.tools_module)
        tool_registry = mod.TOOL_REGISTRY
        tool_schemas  = mod.TOOL_SCHEMAS
        AgentStatus   = mod.AgentStatus

        self._log_gray(f"Tools loaded: {list(tool_registry.keys())}")
        self._log_divider()

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",   "content": self._user_message()},
        ]

        final_result = None

        for iteration in range(1, MAX_ITER + 1):
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
                self._log_green("Agent completed reasoning.")
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
                    result_content = json.dumps({"status": "ESCALATE", "message": f"Unknown tool: {tool_name}"})
                    self._log_red(f"Unknown tool: {tool_name}")
                else:
                    tool_result    = self._invoke_tool(tool_registry[tool_name], args, mod)
                    result_content = tool_result.model_dump_json()

                    self._log_status(tool_result.status.value)
                    self._log_gray(f"  Message       : {tool_result.message[:200]}")

                    # Stream briefing content through logs — the demo wow moment
                    if tool_name == "draft_supervisor_briefing" and tool_result.data.get("briefing"):
                        self._log_divider()
                        self._log_teal("  BRIEFING CONTENT (ready to send):")
                        for line in tool_result.data["briefing"].split("\n"):
                            self._log_gray(f"  {line}")
                        self._log_divider()

                    if tool_result.status == AgentStatus.ESCALATE:
                        self._log_red(f"\nESCALATE — {tool_name} could not complete.")
                        self._log_red(f"Reason: {tool_result.message}")
                        self._log_divider()
                        raise RuntimeError(
                            f"[AirClaw ESCALATE] {tool_name}: {tool_result.message}"
                        )

                    final_result = tool_result

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result_content,
                })
                self._log_divider()

        if final_result is None:
            raise RuntimeError("[AirClaw] Agent loop ended without a result.")

        self._banner("NemoClaw Agent — Triage Complete")
        self._log_green(f"Status  : {final_result.status.value}")
        self._log_green(f"Summary : {final_result.message[:400]}")
        self._log_divider()

        return final_result.model_dump()

    # ── NIM call ───────────────────────────────────────────────────────────────

    def _call_nim(self, api_key, messages, tools):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       NIM_MODEL,
            "messages":    messages,
            "tools":       tools,
            "tool_choice": "auto",
            "max_tokens":  2048,
            "temperature": 0.1,
        }
        try:
            resp = requests.post(
                f"{self.nim_endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            self._log_red(f"NIM API error: {e}")
            return None

    # ── Tool invocation ────────────────────────────────────────────────────────

    def _invoke_tool(self, tool_fn, args, mod):
        import inspect
        sig    = inspect.signature(tool_fn)
        params = list(sig.parameters.values())
        if params:
            input_model = params[0].annotation
            try:
                return tool_fn(input_model(**args))
            except Exception as e:
                return mod.AgentResult(
                    status=mod.AgentStatus.RETRY,
                    message=f"Tool input error: {e}",
                    tool=tool_fn.__name__
                )
        return tool_fn(args)

    # ── Prompts ────────────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return (
            "You are NemoClaw — an autonomous operations agent for NYC 311 service requests. "
            "You triage overnight data, find SLA breaches, detect spikes, draft supervisor "
            "briefings, and surface only what requires human attention.\n\n"
            "Rules:\n"
            "1. validate_schema first — ESCALATE immediately if schema has drifted.\n"
            "2. check_sla_breaches to find overdue cases.\n"
            "3. detect_complaint_spike to find overnight anomalies.\n"
            "4. draft_supervisor_briefing for EACH agency that has breaches — one call per agency.\n"
            "5. generate_summary last — duty manager overview, one paragraph.\n"
            "6. ESCALATE stops the pipeline. RETRY retries once. SUCCESS continues.\n"
            "7. Be decisive. Surface only what needs human attention. Nothing else."
        )

    def _user_message(self) -> str:
        ctx = json.dumps(self.context, indent=2)
        return (
            f"Goal: {self.goal}\n\n"
            f"Context:\n{ctx}\n\n"
            "Begin triage. Start with schema validation."
        )

    # ── Logging ────────────────────────────────────────────────────────────────

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