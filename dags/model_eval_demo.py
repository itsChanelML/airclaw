"""
AirClaw Model Eval DAG — GPT-4o vs Nemotron-Super Migration
------------------------------------------------------------
Three tasks:

  [1] ingest_eval_results → [2] nemoclaw_eval_agent → [3] publish_report

Task 1: Loads the eval CSV, packages context.
Task 2: NemoClawOperator — compares models, detects regressions,
        runs cost analysis, writes migration report.
Task 3: Publishes the report (prints + XCom).

Demo commands:
    Happy path:   cp data/model_eval_clean.csv  data/model_eval_upstream.csv
    Schema drift: cp data/model_eval_broken.csv data/model_eval_upstream.csv
    Then:         airflow dags trigger model_eval_demo
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Airflow parses DAG files in isolation, so make the repo importable regardless
# of whether this file is being read from the repo itself or from a copy in
# AIRFLOW_HOME/dags. Set AIRCLAW_HOME to override.
import sys as _sys
from pathlib import Path as _Path

def _bootstrap_paths():
    import os as _os
    candidates = []
    explicit = _os.environ.get("AIRCLAW_HOME")
    if explicit:
        candidates.append(_Path(explicit).expanduser())
    here = _Path(__file__).resolve()
    candidates.extend([here.parent.parent, *here.parents])
    for base in candidates:
        if (base / "airclaw_env.py").exists() or (base / "tools").is_dir():
            for path in (base, base / "plugins", base / "tools"):
                if path.is_dir() and str(path) not in _sys.path:
                    _sys.path.insert(0, str(path))
            return

_bootstrap_paths()

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from airclaw_env import data_file
from nemoclaw_operator import NemoClawOperator

# Resolved through airclaw_env so the DAG works whether Airflow loads it from
# the repo or from a copy in AIRFLOW_HOME.
UPSTREAM_FILE = str(data_file("model_eval_upstream.csv"))
CLEAN_FILE    = str(data_file("model_eval_clean.csv"))

REQUIRED_FIELDS = [
    "prompt_id", "category", "model_a", "model_b",
    "model_a_quality_score", "model_b_quality_score",
    "model_a_format_score", "model_b_format_score",
    "model_a_latency_sec", "model_b_latency_sec",
    "model_a_cost_usd", "model_b_cost_usd",
    "model_a_refused", "model_b_refused", "regression_flag",
]

AGENT_GOAL = (
    "You are a model migration eval agent. A team is considering migrating "
    "between the two models named in the eval file (model_a -> model_b). "
    "Your job is to do the analysis that would take a Sr Engineer 2-3 days manually "
    "and produce a go/no-go recommendation with evidence. "
    "Step 1: Validate the eval file schema — stop immediately if it has drifted. "
    "Step 2: Run score_comparison to compare model quality by task category. "
    "Step 3: Run detect_regression to find where Model B meaningfully regresses. "
    "Step 4: Run cost_analysis at 100,000 prompts/month to project savings. "
    "Step 5: Run draft_migration_report — pass the full data from all prior tool calls. "
    "Step 6: Run generate_summary with the recommendation. "
    "Be precise. Surface evidence, not opinions. "
    "The VP of Engineering reads this report in 5 minutes and makes a decision."
)

default_args = {
    "owner":            "chanel",
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="model_eval_demo",
    description="AirClaw — Model Migration Eval (GPT-4o vs Nemotron candidate)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["airclaw", "model-eval", "migration", "demo"],
) as dag:

    def ingest_eval_results(**kwargs):
        import csv, shutil
        path = Path(UPSTREAM_FILE)
        if not path.exists():
            shutil.copy(CLEAN_FILE, path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        model_a = rows[0].get("model_a", "Model A") if rows else "Model A"
        model_b = rows[0].get("model_b", "Model B") if rows else "Model B"
        cats    = list({r.get("category","") for r in rows})

        context = {
            "file_path":       str(path.resolve()),
            "required_fields": REQUIRED_FIELDS,
            "total_prompts":   len(rows),
            "model_a":         model_a,
            "model_b":         model_b,
            "categories":      cats,
            "monthly_volume":  100000,
            "as_of":           datetime.now().strftime("%B %d, %Y"),
            "description":     "Production model migration eval — model_a vs model_b from the eval file",
        }

        print(f"[ingest] Eval prompts  : {len(rows)}")
        print(f"[ingest] Model A       : {model_a}")
        print(f"[ingest] Model B       : {model_b}")
        print(f"[ingest] Categories    : {cats}")
        print(f"[ingest] Handing off to NemoClaw eval agent.")

        kwargs["ti"].xcom_push(key="ingest_context", value=context)
        return context

    task_ingest = PythonOperator(task_id="ingest_eval_results", python_callable=ingest_eval_results)

    task_agent = NemoClawOperator(
        task_id="nemoclaw_eval_agent",
        goal=AGENT_GOAL,
        context={
            "file_path":       UPSTREAM_FILE,
            "required_fields": REQUIRED_FIELDS,
            "monthly_volume":  100000,
            "as_of":           datetime.now().strftime("%B %d, %Y"),
            "description":     "Production model migration eval — model_a vs model_b from the eval file",
        },
        tools_module="model_eval_tools",
        nim_api_key_env="NIM_API_KEY",
        max_iterations=14,
    )

    def publish_report(**kwargs):
        ti     = kwargs["ti"]
        result = ti.xcom_pull(task_ids="nemoclaw_eval_agent")

        TEAL  = "\033[38;5;43m"
        GREEN = "\033[38;5;82m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        LINE  = "═" * 62

        print(f"\n{BOLD}{TEAL}{LINE}{RESET}")
        print(f"{BOLD}{TEAL}  AirClaw — Model Migration Report Ready{RESET}")
        print(f"{BOLD}{TEAL}{LINE}{RESET}\n")

        if not result:
            print("[publish] No result from agent.")
            return

        data = result.get("data", {})
        rec  = data.get("recommendation", "")
        color = GREEN if "PROCEED" in rec else TEAL

        print(f"{color}{BOLD}RECOMMENDATION: {rec}{RESET}\n")
        print(result.get("message", ""))
        print(f"\n{TEAL}Full report in XCom. Annual savings: ${data.get('annual_savings', 0):,.0f}{RESET}")
        print(f"\n{BOLD}{TEAL}{LINE}{RESET}")

        return result

    task_publish = PythonOperator(task_id="publish_report", python_callable=publish_report)

    task_ingest >> task_agent >> task_publish