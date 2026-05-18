"""
AirClaw Tool Registry — Model Migration Eval Agent
---------------------------------------------------
These tools do the analysis a Sr Engineer spends 2-3 days
doing manually before a model migration:

  Run both models on production prompts → compare quality,
  latency, cost, refusal rate, format compliance → detect
  regressions → write a go/no-go report with evidence.

The output is a migration recommendation your VP of Eng
can read in 5 minutes and act on. Not a spreadsheet.
Not a notebook. A decision.

Dataset fields:
    prompt_id, category, prompt_preview,
    model_a, model_b,
    model_a_quality_score, model_b_quality_score,
    model_a_format_score,  model_b_format_score,
    model_a_latency_sec,   model_b_latency_sec,
    model_a_cost_usd,      model_b_cost_usd,
    model_a_refused,       model_b_refused,
    regression_flag,       improvement_flag

Python 3.9 compatible.
"""

import csv
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Result contract ────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    SUCCESS  = "SUCCESS"
    RETRY    = "RETRY"
    ESCALATE = "ESCALATE"


class AgentResult(BaseModel):
    status:  AgentStatus
    data:    Dict[str, Any] = {}
    message: str = ""
    tool:    str = ""


# ── Input schemas ──────────────────────────────────────────────────────────────

class ValidateSchemaInput(BaseModel):
    file_path:       str
    required_fields: List[str]


class ScoreComparisonInput(BaseModel):
    file_path:  str
    categories: List[str] = []   # empty = all categories


class DetectRegressionInput(BaseModel):
    file_path:         str
    min_score_drop:    int   = 10    # flag if Model B drops this many points
    refusal_threshold: float = 0.10  # flag if Model B refuses >10% more


class CostAnalysisInput(BaseModel):
    file_path:    str
    monthly_volume: int = 100000   # estimated monthly prompt volume


class DraftMigrationReportInput(BaseModel):
    model_a:          str
    model_b:          str
    scores:           Dict[str, Any] = {}
    regressions:      Dict[str, Any] = {}
    cost_analysis:    Dict[str, Any] = {}
    evaluated_on:     int = 0
    as_of:            str = ""


class GenerateSummaryInput(BaseModel):
    recommendation:  str = ""
    model_a:         str = ""
    model_b:         str = ""
    total_evaluated: int = 0
    original_goal:   str = ""


# ── Tools ──────────────────────────────────────────────────────────────────────

def validate_schema(input: ValidateSchemaInput) -> AgentResult:
    """
    Verify the eval file has all required fields.
    ESCALATE with exact diagnosis if schema has drifted —
    the broken file renames model_b_quality_score → model_b_score.
    """
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message=f"Eval file not found: {input.file_path}. Cannot run migration analysis.",
            tool="validate_schema"
        )

    with open(path, newline="") as f:
        actual_fields = csv.DictReader(f).fieldnames or []

    missing = [field for field in input.required_fields if field not in actual_fields]

    if missing:
        suggestions = {}
        for m in missing:
            candidates = [a for a in actual_fields if m in a or a in m]
            if candidates:
                suggestions[m] = candidates[0]
        diag = (
            f"Schema drift in eval file. "
            f"Missing required fields: {missing}. "
        )
        if suggestions:
            diag += (
                f"Likely renames by upstream eval pipeline: {suggestions}. "
                f"Migration report cannot be generated until schema is corrected."
            )
        diag += f" Fields received: {list(actual_fields)}."
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message=diag,
            data={"missing": missing, "actual": list(actual_fields), "suggestions": suggestions},
            tool="validate_schema"
        )

    with open(path, newline="") as f:
        row_count = sum(1 for _ in csv.DictReader(f))

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=f"Schema valid. {row_count} eval prompts ready for analysis.",
        data={"fields": list(actual_fields), "row_count": row_count},
        tool="validate_schema"
    )


def score_comparison(input: ScoreComparisonInput) -> AgentResult:
    """
    Compare Model A vs Model B quality and format scores
    broken down by task category. Surfaces where each model
    wins, loses, and by how much.
    """
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(status=AgentStatus.ESCALATE, message=f"File not found: {input.file_path}", tool="score_comparison")

    by_category: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    model_a_name = ""
    model_b_name = ""
    total = 0

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cat = row.get("category", "unknown")
                if input.categories and cat not in input.categories:
                    continue
                if not model_a_name:
                    model_a_name = row.get("model_a", "Model A")
                    model_b_name = row.get("model_b", "Model B")
                try:
                    by_category[cat]["a_quality"].append(float(row.get("model_a_quality_score", 0)))
                    by_category[cat]["b_quality"].append(float(row.get("model_b_quality_score", 0)))
                    by_category[cat]["a_format"].append(float(row.get("model_a_format_score", 0)))
                    by_category[cat]["b_format"].append(float(row.get("model_b_format_score", 0)))
                    total += 1
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return AgentResult(status=AgentStatus.RETRY, message=f"Parse error: {e}", tool="score_comparison")

    results = {}
    overall_a, overall_b = [], []

    for cat, scores in by_category.items():
        avg_aq = round(sum(scores["a_quality"]) / len(scores["a_quality"]), 1)
        avg_bq = round(sum(scores["b_quality"]) / len(scores["b_quality"]), 1)
        avg_af = round(sum(scores["a_format"])  / len(scores["a_format"]),  1)
        avg_bf = round(sum(scores["b_format"])  / len(scores["b_format"]),  1)
        delta  = round(avg_bq - avg_aq, 1)
        winner = "Model B" if delta > 2 else ("Model A" if delta < -2 else "Tie")

        results[cat] = {
            "model_a_quality": avg_aq,
            "model_b_quality": avg_bq,
            "model_a_format":  avg_af,
            "model_b_format":  avg_bf,
            "quality_delta":   delta,
            "winner":          winner,
            "sample_count":    len(scores["a_quality"]),
        }
        overall_a.extend(scores["a_quality"])
        overall_b.extend(scores["b_quality"])

    overall_delta = round(
        sum(overall_b) / len(overall_b) - sum(overall_a) / len(overall_a), 1
    ) if overall_a and overall_b else 0

    b_wins = [c for c, r in results.items() if r["winner"] == "Model B"]
    a_wins = [c for c, r in results.items() if r["winner"] == "Model A"]

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"Score comparison complete across {total} prompts and {len(results)} categories. "
            f"Overall quality delta: {'+' if overall_delta > 0 else ''}{overall_delta} pts in favor of "
            f"{'Model B' if overall_delta > 0 else 'Model A'}. "
            f"Model B wins on: {b_wins}. Model A wins on: {a_wins}."
        ),
        data={
            "by_category":     results,
            "overall_delta":   overall_delta,
            "model_b_wins":    b_wins,
            "model_a_wins":    a_wins,
            "model_a_name":    model_a_name,
            "model_b_name":    model_b_name,
            "total_evaluated": total,
        },
        tool="score_comparison"
    )


def detect_regression(input: DetectRegressionInput) -> AgentResult:
    """
    Find categories where Model B meaningfully regresses —
    not just scores lower, but drops enough to matter in production.
    Also flags if Model B's refusal rate spiked, which breaks
    user-facing features that depend on the model always responding.
    """
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(status=AgentStatus.ESCALATE, message=f"File not found: {input.file_path}", tool="detect_regression")

    by_category: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    model_a_name = ""
    model_b_name = ""

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cat = row.get("category", "unknown")
                if not model_a_name:
                    model_a_name = row.get("model_a", "Model A")
                    model_b_name = row.get("model_b", "Model B")
                try:
                    by_category[cat]["a_q"].append(float(row.get("model_a_quality_score", 0)))
                    by_category[cat]["b_q"].append(float(row.get("model_b_quality_score", 0)))
                    by_category[cat]["a_refused"].append(1 if row.get("model_a_refused") == "YES" else 0)
                    by_category[cat]["b_refused"].append(1 if row.get("model_b_refused") == "YES" else 0)
                    by_category[cat]["regression"].append(1 if row.get("regression_flag") == "YES" else 0)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return AgentResult(status=AgentStatus.RETRY, message=f"Parse error: {e}", tool="detect_regression")

    regressions = []
    refusal_spikes = []

    for cat, data in by_category.items():
        avg_a = sum(data["a_q"]) / len(data["a_q"])
        avg_b = sum(data["b_q"]) / len(data["b_q"])
        drop  = round(avg_a - avg_b, 1)

        refusal_a = sum(data["a_refused"]) / len(data["a_refused"])
        refusal_b = sum(data["b_refused"]) / len(data["b_refused"])
        refusal_increase = round(refusal_b - refusal_a, 3)

        regression_count = sum(data["regression"])
        regression_pct   = round(regression_count / len(data["a_q"]) * 100, 1)

        if drop >= input.min_score_drop:
            regressions.append({
                "category":        cat,
                "model_a_avg":     round(avg_a, 1),
                "model_b_avg":     round(avg_b, 1),
                "quality_drop":    drop,
                "regression_pct":  regression_pct,
                "severity":        "HIGH" if drop >= 20 else "MODERATE",
                "recommendation":  (
                    f"Do NOT migrate {cat} to {model_b_name} without fine-tuning. "
                    f"Quality drop of {drop} pts affects {regression_pct}% of prompts in this category."
                ),
            })

        if refusal_increase >= input.refusal_threshold:
            refusal_spikes.append({
                "category":          cat,
                "model_a_refusal":   f"{round(refusal_a * 100, 1)}%",
                "model_b_refusal":   f"{round(refusal_b * 100, 1)}%",
                "increase":          f"+{round(refusal_increase * 100, 1)}pp",
                "severity":          "HIGH" if refusal_increase >= 0.20 else "MODERATE",
                "recommendation":    (
                    f"Model B refuses {round(refusal_increase*100, 1)}pp more often on {cat} prompts. "
                    f"User-facing features in this category will see increased no-answer rates."
                ),
            })

    all_issues = regressions + refusal_spikes
    if not all_issues:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message="No significant regressions detected. Model B performs within acceptable range across all categories.",
            data={"regressions": [], "refusal_spikes": [], "clear_to_migrate": True},
            tool="detect_regression"
        )

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"{len(regressions)} quality regression(s) and {len(refusal_spikes)} refusal spike(s) detected. "
            f"Categories with issues: "
            f"{list({r['category'] for r in all_issues})}. "
            f"Migration should be partial — hold impacted categories."
        ),
        data={
            "regressions":      regressions,
            "refusal_spikes":   refusal_spikes,
            "clear_to_migrate": len(all_issues) == 0,
            "hold_categories":  list({r["category"] for r in all_issues}),
        },
        tool="detect_regression"
    )


def cost_analysis(input: CostAnalysisInput) -> AgentResult:
    """
    Compute cost and latency comparison between Model A and B.
    Projects monthly and annual savings at estimated volume.
    This is the number that gets the migration approved.
    """
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(status=AgentStatus.ESCALATE, message=f"File not found: {input.file_path}", tool="cost_analysis")

    costs_a, costs_b, latencies_a, latencies_b = [], [], [], []
    model_a_name = ""
    model_b_name = ""

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not model_a_name:
                    model_a_name = row.get("model_a", "Model A")
                    model_b_name = row.get("model_b", "Model B")
                try:
                    costs_a.append(float(row.get("model_a_cost_usd", 0)))
                    costs_b.append(float(row.get("model_b_cost_usd", 0)))
                    latencies_a.append(float(row.get("model_a_latency_sec", 0)))
                    latencies_b.append(float(row.get("model_b_latency_sec", 0)))
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return AgentResult(status=AgentStatus.RETRY, message=f"Parse error: {e}", tool="cost_analysis")

    avg_cost_a   = sum(costs_a)    / len(costs_a)
    avg_cost_b   = sum(costs_b)    / len(costs_b)
    avg_lat_a    = sum(latencies_a)/ len(latencies_a)
    avg_lat_b    = sum(latencies_b)/ len(latencies_b)

    cost_savings_pct  = round((1 - avg_cost_b / avg_cost_a) * 100, 1)
    latency_delta_pct = round((1 - avg_lat_b  / avg_lat_a)  * 100, 1)

    monthly_cost_a    = round(avg_cost_a * input.monthly_volume, 2)
    monthly_cost_b    = round(avg_cost_b * input.monthly_volume, 2)
    monthly_savings   = round(monthly_cost_a - monthly_cost_b, 2)
    annual_savings    = round(monthly_savings * 12, 2)

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"Cost analysis complete across {len(costs_a)} prompts at "
            f"{input.monthly_volume:,} prompts/month estimated volume. "
            f"{model_b_name} is {cost_savings_pct}% cheaper per prompt "
            f"and {latency_delta_pct}% faster. "
            f"Projected annual savings at current volume: ${annual_savings:,.0f}."
        ),
        data={
            "model_a_name":       model_a_name,
            "model_b_name":       model_b_name,
            "avg_cost_a":         round(avg_cost_a, 5),
            "avg_cost_b":         round(avg_cost_b, 5),
            "avg_latency_a_sec":  round(avg_lat_a, 2),
            "avg_latency_b_sec":  round(avg_lat_b, 2),
            "cost_savings_pct":   cost_savings_pct,
            "latency_delta_pct":  latency_delta_pct,
            "monthly_cost_a":     monthly_cost_a,
            "monthly_cost_b":     monthly_cost_b,
            "monthly_savings":    monthly_savings,
            "annual_savings":     annual_savings,
            "monthly_volume":     input.monthly_volume,
        },
        tool="cost_analysis"
    )


def draft_migration_report(input: DraftMigrationReportInput) -> AgentResult:
    """
    Write a go/no-go migration recommendation with evidence.
    VP of Engineering reads this in 5 minutes and makes a decision.
    Not a spreadsheet. Not a notebook. A decision document.
    """
    if not input.model_a or not input.model_b:
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message="Model A and Model B names required to draft migration report.",
            tool="draft_migration_report"
        )

    as_of     = input.as_of or datetime.now().strftime("%B %d, %Y")
    scores    = input.scores
    regs      = input.regressions
    costs     = input.cost_analysis
    evaluated = input.evaluated_on
    lines     = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"MODEL MIGRATION REPORT")
    lines.append(f"{input.model_a} → {input.model_b}")
    lines.append(f"Evaluated on {evaluated:,} production prompts | {as_of}")
    lines.append(f"Generated by AirClaw Eval Agent")
    lines.append("")

    # ── Recommendation ────────────────────────────────────────────────────────
    hold_cats   = regs.get("hold_categories", [])
    clear_cats  = [c for c in scores.get("model_b_wins", []) if c not in hold_cats]
    all_clear   = regs.get("clear_to_migrate", False)

    if all_clear:
        rec = "PROCEED — Full migration recommended."
        rec_detail = (
            f"{input.model_b} performs at or above {input.model_a} across all categories "
            f"with no significant regressions detected."
        )
    elif clear_cats and hold_cats:
        rec = f"PROCEED PARTIALLY — Migrate {', '.join(clear_cats)}. Hold {', '.join(hold_cats)}."
        rec_detail = (
            f"{input.model_b} meaningfully outperforms {input.model_a} on {', '.join(clear_cats)}. "
            f"Regressions detected on {', '.join(hold_cats)} — hold these until fine-tuning or "
            f"prompt engineering resolves the gaps."
        )
    else:
        rec = f"HOLD — Do not migrate until regressions on {', '.join(hold_cats)} are resolved."
        rec_detail = (
            f"Quality drops on critical task categories make a full migration premature. "
            f"Address regressions before proceeding."
        )

    lines.append(f"RECOMMENDATION: {rec}")
    lines.append("")
    lines.append(rec_detail)
    lines.append("")

    # ── Quality comparison ─────────────────────────────────────────────────────
    lines.append("QUALITY COMPARISON BY CATEGORY:")
    lines.append("")
    by_cat = scores.get("by_category", {})
    for cat, data in by_cat.items():
        delta  = data.get("quality_delta", 0)
        winner = data.get("winner", "Tie")
        sign   = "+" if delta > 0 else ""
        lines.append(f"  {cat.replace('_', ' ').title()}")
        lines.append(f"    {input.model_a}: {data.get('model_a_quality', 0)} | {input.model_b}: {data.get('model_b_quality', 0)} | Delta: {sign}{delta} pts | {winner}")
        lines.append("")

    # ── Regressions ───────────────────────────────────────────────────────────
    reg_list = regs.get("regressions", [])
    ref_list = regs.get("refusal_spikes", [])

    if reg_list or ref_list:
        lines.append("REGRESSIONS DETECTED:")
        lines.append("")
        for r in reg_list:
            lines.append(f"  ✗ {r['category'].replace('_', ' ').title()} — Quality drop: {r['quality_drop']} pts ({r['severity']})")
            lines.append(f"    {r['recommendation']}")
            lines.append("")
        for r in ref_list:
            lines.append(f"  ✗ {r['category'].replace('_', ' ').title()} — Refusal rate: {r['model_a_refusal']} → {r['model_b_refusal']} ({r['increase']}, {r['severity']})")
            lines.append(f"    {r['recommendation']}")
            lines.append("")
    else:
        lines.append("REGRESSIONS: None detected.")
        lines.append("")

    # ── Cost and latency ──────────────────────────────────────────────────────
    if costs:
        lines.append("COST AND LATENCY:")
        lines.append("")
        lines.append(f"  Cost per prompt   : ${costs.get('avg_cost_a', 0):.5f} → ${costs.get('avg_cost_b', 0):.5f} ({costs.get('cost_savings_pct', 0)}% reduction)")
        lines.append(f"  Avg latency       : {costs.get('avg_latency_a_sec', 0)}s → {costs.get('avg_latency_b_sec', 0)}s ({costs.get('latency_delta_pct', 0)}% faster)")
        lines.append(f"  Monthly savings   : ${costs.get('monthly_savings', 0):,.0f} at {costs.get('monthly_volume', 0):,} prompts/month")
        lines.append(f"  Annual projection : ${costs.get('annual_savings', 0):,.0f}")
        lines.append("")

    # ── Next steps ────────────────────────────────────────────────────────────
    lines.append("RECOMMENDED NEXT STEPS:")
    lines.append("")
    if clear_cats:
        lines.append(f"  1. Route {', '.join(clear_cats)} traffic to {input.model_b} — no further action needed.")
    if hold_cats:
        lines.append(f"  2. Begin prompt engineering or fine-tuning for {', '.join(hold_cats)}.")
        lines.append(f"  3. Re-run this eval pipeline in 30 days after changes are applied.")
    if all_clear:
        lines.append(f"  1. Proceed with full migration. Update model config and monitor for 7 days.")
        lines.append(f"  2. Set up automated weekly eval runs to catch future drift early.")

    lines.append("")
    lines.append("—")
    lines.append("AirClaw Eval Agent | Automated Model Migration Analysis")
    lines.append("This report was generated automatically from production eval data.")

    report_text = "\n".join(lines)

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"Migration report drafted: {rec.split(' — ')[0]}. "
            f"Evaluated {evaluated:,} prompts. "
            f"{len(reg_list)} regression(s), {len(ref_list)} refusal spike(s) detected. "
            f"Annual cost savings if migrated: ${costs.get('annual_savings', 0):,.0f}."
        ),
        data={
            "recommendation":  rec,
            "clear_categories": clear_cats,
            "hold_categories":  hold_cats,
            "report":           report_text,
            "annual_savings":   costs.get("annual_savings", 0),
        },
        tool="draft_migration_report"
    )


def generate_summary(input: GenerateSummaryInput) -> AgentResult:
    lines = [
        f"AirClaw eval run complete — {datetime.now().strftime('%Y-%m-%d %H:%M')}."
    ]
    if input.total_evaluated:
        lines.append(f"Evaluated {input.total_evaluated:,} prompts comparing {input.model_a} vs {input.model_b}.")
    if input.recommendation:
        lines.append(f"Recommendation: {input.recommendation}.")
    lines.append("Migration report ready for engineering and leadership review.")
    summary = " ".join(lines)
    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=summary,
        data={"summary": summary, "recommendation": input.recommendation},
        tool="generate_summary"
    )


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "validate_schema":       validate_schema,
    "score_comparison":      score_comparison,
    "detect_regression":     detect_regression,
    "cost_analysis":         cost_analysis,
    "draft_migration_report": draft_migration_report,
    "generate_summary":      generate_summary,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "validate_schema",
            "description": "Verify the eval CSV has all required fields. Always call first. ESCALATE with exact diagnosis if schema has drifted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":       {"type": "string"},
                    "required_fields": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["file_path", "required_fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "score_comparison",
            "description": "Compare Model A vs Model B quality and format scores broken down by task category. Returns which model wins on each category and by how much. Call after validate_schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":  {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific categories. Empty = all."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_regression",
            "description": "Find categories where Model B meaningfully drops in quality or significantly increases refusal rate. These categories should be held from migration. Call after score_comparison.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":         {"type": "string"},
                    "min_score_drop":    {"type": "integer", "description": "Minimum quality drop to flag as regression (default 10 pts)"},
                    "refusal_threshold": {"type": "number",  "description": "Minimum increase in refusal rate to flag (default 0.10 = 10pp)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cost_analysis",
            "description": "Compute cost per prompt and latency for both models. Projects monthly and annual savings at estimated volume. This is the number that gets the migration approved. Call after detect_regression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":       {"type": "string"},
                    "monthly_volume":  {"type": "integer", "description": "Estimated monthly prompt volume for cost projection (default 100000)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_migration_report",
            "description": "Write a go/no-go migration recommendation with evidence from all prior tool calls. VP of Engineering reads this in 5 minutes and makes a decision. Call after cost_analysis. Pass the full data from score_comparison, detect_regression, and cost_analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_a":        {"type": "string", "description": "Name of the current model e.g. gpt-4o"},
                    "model_b":        {"type": "string", "description": "Name of the candidate model"},
                    "scores":         {"type": "object", "description": "Full data dict from score_comparison"},
                    "regressions":    {"type": "object", "description": "Full data dict from detect_regression"},
                    "cost_analysis":  {"type": "object", "description": "Full data dict from cost_analysis"},
                    "evaluated_on":   {"type": "integer", "description": "Total number of prompts evaluated"},
                    "as_of":          {"type": "string"}
                },
                "required": ["model_a", "model_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary",
            "description": "One-paragraph summary of the eval run. Call last — final XCom output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recommendation":  {"type": "string"},
                    "model_a":         {"type": "string"},
                    "model_b":         {"type": "string"},
                    "total_evaluated": {"type": "integer"},
                    "original_goal":   {"type": "string"}
                },
                "required": []
            }
        }
    }
]