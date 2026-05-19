"""
AirClaw Tool Registry — NYC 311 Productivity Agent
---------------------------------------------------
These tools do the work a city agency analyst spends
30-60 minutes doing manually every morning.

Tools:
  1. validate_schema           — verify upstream data is clean
  2. check_sla_breaches        — find every overdue open request
  3. detect_complaint_spike    — flag overnight volume anomalies
  4. prioritize_queue          — rank by severity, cluster by geography
  5. draft_supervisor_briefing — write actionable ready-to-send briefing
  6. generate_summary          — duty manager one-pager

Python 3.9 compatible.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
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


# ── Complaint severity — higher = more urgent ──────────────────────────────────

COMPLAINT_SEVERITY = {
    "HEAT/HOT WATER":                      10,
    "Rodent":                               9,
    "UNSANITARY CONDITION":                 9,
    "Homeless Person Assistance":           9,
    "Elevator":                             8,
    "PLUMBING":                             8,
    "PAINT/PLASTER":                        7,
    "Water System":                         7,
    "Street Light Condition":               6,
    "Street Condition":                     6,
    "Noise - Residential":                  5,
    "Noise - Commercial":                   5,
    "Noise - Street/Sidewalk":              5,
    "Noise - Vehicle":                      4,
    "Blocked Driveway":                     4,
    "Illegal Parking":                      4,
    "Derelict Vehicle":                     3,
    "Graffiti":                             3,
    "Request Large Bulky Item Collection":  2,
    "Non-Emergency Police Matter":          2,
}


# ── Input schemas ──────────────────────────────────────────────────────────────

class ValidateSchemaInput(BaseModel):
    file_path:       str
    required_fields: List[str]


class SLABreachInput(BaseModel):
    file_path:     str
    agency_filter: str = ""
    hours_window:  int = 24


class SpikeDetectionInput(BaseModel):
    file_path:       str
    baseline_days:   int   = 7
    spike_threshold: float = 0.40
    hours_window:    int   = 24


class PrioritizeQueueInput(BaseModel):
    agency:   str
    breaches: List[Dict[str, Any]]
    spikes:   List[Dict[str, Any]] = []


class DraftBriefingInput(BaseModel):
    agency:      str
    supervisor:  str
    file_path:   str
    spikes:      List[Dict[str, Any]] = []
    as_of:       str = ""


class GenerateSummaryInput(BaseModel):
    all_breaches:    List[Dict[str, Any]] = []
    all_spikes:      List[Dict[str, Any]] = []
    briefings:       List[str]            = []
    overnight_total: int = 0
    original_goal:   str = ""


# ── Tools ──────────────────────────────────────────────────────────────────────

def validate_schema(input: ValidateSchemaInput) -> AgentResult:
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message=f"Upstream file not found: {input.file_path}. Pipeline cannot proceed.",
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
            f"Schema drift detected in upstream 311 feed. "
            f"Missing required fields: {missing}. "
        )
        if suggestions:
            diag += f"Likely renames by upstream team: {suggestions}. "
        diag += f"Fields received: {list(actual_fields)}."
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
        message=f"Schema valid. {row_count} service requests ready for processing.",
        data={"fields": list(actual_fields), "row_count": row_count},
        tool="validate_schema"
    )


def check_sla_breaches(input: SLABreachInput) -> AgentResult:
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(status=AgentStatus.ESCALATE, message=f"File not found: {input.file_path}", tool="check_sla_breaches")

    breaches = []
    by_agency = defaultdict(list)

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("sla_breach", "").strip() != "YES":
                    continue
                if input.agency_filter and row.get("agency", "") != input.agency_filter:
                    continue
                hours_open = row.get("hours_open", "")
                sla_hours  = row.get("sla_hours", "")
                overdue_by = ""
                if hours_open and sla_hours:
                    try:
                        overdue_by = f"{round(float(hours_open) - float(sla_hours), 1)}h overdue"
                    except ValueError:
                        pass
                record = {
                    "case_id":        row.get("unique_key", ""),
                    "complaint_type": row.get("complaint_type", ""),
                    "borough":        row.get("borough", ""),
                    "district":       row.get("district", ""),
                    "agency":         row.get("agency", ""),
                    "supervisor":     row.get("supervisor", ""),
                    "created_date":   row.get("created_date", ""),
                    "hours_open":     hours_open,
                    "sla_hours":      sla_hours,
                    "overdue_by":     overdue_by,
                }
                breaches.append(record)
                by_agency[row.get("agency", "UNKNOWN")].append(record)
    except Exception as e:
        return AgentResult(status=AgentStatus.RETRY, message=f"Error reading data: {e}", tool="check_sla_breaches")

    if not breaches:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message="No SLA breaches found. All open requests within response windows.",
            data={"breaches": [], "by_agency": {}, "total": 0},
            tool="check_sla_breaches"
        )

    agency_summary = {}
    for agency, cases in by_agency.items():
        worst = sorted([c for c in cases if c["hours_open"]], key=lambda x: float(x["hours_open"]) if x["hours_open"] else 0, reverse=True)
        agency_summary[agency] = {
            "count":      len(cases),
            "supervisor": cases[0]["supervisor"] if cases else "",
            "worst_case": worst[0] if worst else cases[0],
            "cases":      cases,
        }

    top_agency = max(agency_summary.items(), key=lambda x: x[1]["count"])
    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"{len(breaches)} open request(s) have exceeded their SLA window. "
            f"Highest backlog: {top_agency[0]} with {top_agency[1]['count']} breaches "
            f"(supervisor: {top_agency[1]['supervisor']}). "
            f"Agencies affected: {list(agency_summary.keys())}."
        ),
        data={"breaches": breaches, "by_agency": agency_summary, "total": len(breaches), "agencies": list(agency_summary.keys())},
        tool="check_sla_breaches"
    )


def detect_complaint_spike(input: SpikeDetectionInput) -> AgentResult:
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(status=AgentStatus.ESCALATE, message=f"File not found: {input.file_path}", tool="detect_complaint_spike")

    now            = datetime.now()
    overnight_cut  = now - timedelta(hours=input.hours_window)
    baseline_start = now - timedelta(days=input.baseline_days)
    overnight_counts = defaultdict(int)
    baseline_counts  = defaultdict(lambda: [0] * input.baseline_days)

    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                ct  = row.get("complaint_type", "UNKNOWN")
                raw = row.get("created_date", "")
                if not raw:
                    continue
                try:
                    dt = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    continue
                if dt >= overnight_cut:
                    overnight_counts[ct] += 1
                elif dt >= baseline_start:
                    day_idx = (dt - baseline_start).days
                    if 0 <= day_idx < input.baseline_days:
                        baseline_counts[ct][day_idx] += 1
    except Exception as e:
        return AgentResult(status=AgentStatus.RETRY, message=f"Parse error: {e}", tool="detect_complaint_spike")

    spikes = []
    for ct, overnight_vol in overnight_counts.items():
        baseline_daily = baseline_counts.get(ct, [0] * input.baseline_days)
        avg_daily      = sum(baseline_daily) / max(len(baseline_daily), 1)
        if avg_daily == 0:
            avg_daily = 1
        increase = (overnight_vol - avg_daily) / avg_daily
        if increase >= input.spike_threshold:
            spikes.append({
                "complaint_type":  ct,
                "overnight_count": overnight_vol,
                "baseline_avg":    round(avg_daily, 1),
                "increase_pct":    f"{round(increase * 100)}%",
                "severity":        "HIGH" if increase >= 1.0 else "MODERATE",
            })

    spikes.sort(key=lambda x: float(x["increase_pct"].strip("%")), reverse=True)

    if not spikes:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message="No complaint spikes detected. Overnight volume within normal range.",
            data={"spikes": [], "overnight_total": sum(overnight_counts.values())},
            tool="detect_complaint_spike"
        )

    top = spikes[0]
    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"{len(spikes)} complaint type(s) spiked overnight. "
            f"Largest: {top['complaint_type']} — {top['overnight_count']} requests "
            f"vs baseline avg {top['baseline_avg']}/day ({top['increase_pct']} increase)."
        ),
        data={"spikes": spikes, "overnight_total": sum(overnight_counts.values())},
        tool="detect_complaint_spike"
    )


def prioritize_queue(input: PrioritizeQueueInput) -> AgentResult:
    """
    Rank breaches by urgency, cluster by geography.
    Output feeds directly into draft_supervisor_briefing.
    """
    if not input.breaches:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message=f"No breaches to prioritize for {input.agency}.",
            data={"ranked": [], "clusters": {}, "agency": input.agency},
            tool="prioritize_queue"
        )

    max_hours = max(
        (float(b["hours_open"]) for b in input.breaches if b.get("hours_open")),
        default=1
    )

    scored = []
    for b in input.breaches:
        hours = float(b.get("hours_open", 0)) if b.get("hours_open") else 0
        sla   = float(b.get("sla_hours",  1)) if b.get("sla_hours")  else 1
        time_score     = hours / max(max_hours, 1)
        severity_score = COMPLAINT_SEVERITY.get(b.get("complaint_type", ""), 3) / 10
        overdue_ratio  = (hours - sla) / max(sla, 1)
        composite      = (time_score * 0.5) + (severity_score * 0.3) + (min(overdue_ratio, 1) * 0.2)
        scored.append({**b, "_score": round(composite, 4)})

    ranked = sorted(scored, key=lambda x: x["_score"], reverse=True)

    clusters = defaultdict(list)
    for b in ranked:
        clusters[b.get("district", "Unknown")].append(b)

    action_items = []
    seen = set()

    for item in ranked:
        cid = item.get("case_id", "")
        if cid in seen:
            continue
        seen.add(cid)

        ct       = item.get("complaint_type", "")
        district = item.get("district", "")
        borough  = item.get("borough", "")
        overdue  = item.get("overdue_by", "overdue")
        hours    = item.get("hours_open", "")
        severity = COMPLAINT_SEVERITY.get(ct, 3)

        district_cases = [
            b for b in clusters.get(district, [])
            if b.get("case_id", "") != cid and b.get("case_id", "") not in seen
        ]
        cluster_note = ""
        if district_cases:
            cluster_note = (
                f"{len(district_cases)} other open case(s) in {district} — "
                f"single dispatch could cover all {1 + len(district_cases)}."
            )
            for c in district_cases:
                seen.add(c.get("case_id", ""))

        if severity >= 9:
            action = "Immediate field assignment required — health/safety risk."
        elif severity >= 7:
            action = "Assign today. Owner notification required before close of business."
        elif severity >= 5:
            action = "Queue for next available field unit."
        else:
            action = "Schedule during next routine patrol."

        action_items.append({
            "rank":         len(action_items) + 1,
            "case_id":      cid,
            "complaint":    ct,
            "location":     f"{district}, {borough}",
            "overdue":      overdue,
            "hours_open":   hours,
            "severity":     severity,
            "action":       action,
            "cluster_note": cluster_note,
        })

    top = action_items[0] if action_items else {}
    cluster_count = sum(1 for a in action_items if a.get("cluster_note"))

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"Queue prioritized for {input.agency}. "
            f"{len(action_items)} action item(s) ranked. "
            f"Top priority: {top.get('complaint', '')} in {top.get('location', '')} "
            f"({top.get('overdue', '')}). "
            f"{cluster_count} dispatch cluster(s) identified."
        ),
        data={
            "agency":        input.agency,
            "ranked":        action_items,
            "clusters":      {k: len(v) for k, v in clusters.items() if len(v) > 1},
            "total_actions": len(action_items),
        },
        tool="prioritize_queue"
    )


def draft_supervisor_briefing(input: DraftBriefingInput) -> AgentResult:
    """
    Reads SLA breaches for the given agency directly from file_path,
    prioritizes by severity and hours overdue, clusters by geography,
    and writes a ready-to-send morning briefing.
    Call with: agency, supervisor, file_path. That is all.
    """
    if not input.agency or not input.supervisor:
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message="Agency and supervisor name required to draft a briefing.",
            tool="draft_supervisor_briefing"
        )

    import csv as _csv
    from pathlib import Path as _Path
    from collections import defaultdict as _dd

    # ── Load breaches for this agency from file ────────────────────────────────
    path = _Path(input.file_path)
    raw_breaches = []
    if path.exists():
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("sla_breach","").strip() != "YES":
                    continue
                if row.get("agency","").strip() != input.agency:
                    continue
                hours_open = row.get("hours_open","")
                sla_hours  = row.get("sla_hours","")
                overdue_by = ""
                if hours_open and sla_hours:
                    try:
                        overdue_by = f"{round(float(hours_open)-float(sla_hours),1)}h overdue"
                    except ValueError:
                        pass
                raw_breaches.append({
                    "case_id":        row.get("unique_key",""),
                    "complaint_type": row.get("complaint_type",""),
                    "borough":        row.get("borough",""),
                    "district":       row.get("district",""),
                    "hours_open":     hours_open,
                    "sla_hours":      sla_hours,
                    "overdue_by":     overdue_by,
                })

    # ── Prioritize by composite score ──────────────────────────────────────────
    max_hours = max((float(b["hours_open"]) for b in raw_breaches if b.get("hours_open")), default=1)
    scored = []
    for b in raw_breaches:
        hours    = float(b.get("hours_open",0)) if b.get("hours_open") else 0
        sla      = float(b.get("sla_hours",1))  if b.get("sla_hours")  else 1
        ts       = hours / max(max_hours,1)
        ss       = COMPLAINT_SEVERITY.get(b.get("complaint_type",""),3) / 10
        oratio   = (hours - sla) / max(sla,1)
        composite= (ts*0.5) + (ss*0.3) + (min(oratio,1)*0.2)
        scored.append({**b, "_score": composite})
    ranked_raw = sorted(scored, key=lambda x: x["_score"], reverse=True)

    # ── Cluster by district ────────────────────────────────────────────────────
    clusters = _dd(list)
    for b in ranked_raw:
        clusters[b.get("district","Unknown")].append(b)

    queue = []
    seen  = set()
    for item in ranked_raw:
        cid = item.get("case_id","")
        if cid in seen:
            continue
        seen.add(cid)
        ct       = item.get("complaint_type","")
        district = item.get("district","")
        borough  = item.get("borough","")
        overdue  = item.get("overdue_by","overdue")
        severity = COMPLAINT_SEVERITY.get(ct,3)
        others   = [b for b in clusters.get(district,[])
                    if b.get("case_id","") != cid and b.get("case_id","") not in seen]
        cluster_note = ""
        if others:
            cluster_note = (f"{len(others)} other open case(s) in {district} — "
                            f"single dispatch could cover all {1+len(others)}.")
            for c in others:
                seen.add(c.get("case_id",""))
        if severity >= 9:   action = "Immediate field assignment required — health/safety risk."
        elif severity >= 7: action = "Assign today. Owner notification required before close of business."
        elif severity >= 5: action = "Queue for next available field unit."
        else:               action = "Schedule during next routine patrol."
        queue.append({
            "rank":         len(queue)+1,
            "case_id":      cid,
            "complaint":    ct,
            "location":     f"{district}, {borough}",
            "overdue":      overdue,
            "severity":     severity,
            "action":       action,
            "cluster_note": cluster_note,
        })

    # ── Write the briefing ─────────────────────────────────────────────────────
    as_of  = input.as_of or datetime.now().strftime("%B %d, %Y at %I:%M %p")
    spikes = input.spikes
    lines  = []

    lines.append(f"SUBJECT: Morning Ops Briefing — {input.agency} | {as_of}")
    lines.append(f"TO: {input.supervisor}")
    lines.append("")

    if not queue:
        lines.append(f"Good morning — your queue is clear. No SLA breaches as of {as_of}.")
    else:
        lines.append(
            f"Good morning — you have {len(queue)} open request(s) past their SLA window "
            f"as of {as_of}. Here is what needs your attention, ranked by priority."
        )
    lines.append("")

    if queue:
        lines.append("PRIORITY ACTIONS:")
        lines.append("")
        for item in queue[:7]:
            lines.append(f"  {item['rank']}. {item['complaint']} — {item['location']}")
            lines.append(f"     Case: {item['case_id']} | {item['overdue']}")
            lines.append(f"     → {item['action']}")
            if item.get("cluster_note"):
                lines.append(f"     ℹ {item['cluster_note']}")
            lines.append("")
        if len(queue) > 7:
            lines.append(f"  + {len(queue)-7} additional breach(es) below priority threshold.")
            lines.append("")

    if spikes:
        lines.append("OVERNIGHT SPIKE ALERTS:")
        lines.append("")
        for s in spikes[:3]:
            lines.append(f"  • {s.get('complaint_type','')}: {s.get('overnight_count','')} overnight "
                         f"vs avg {s.get('baseline_avg','')}/day ({s.get('increase_pct','')} above baseline)")
        lines.append("")
        lines.append("  → Monitor for continued increase. Consider pre-positioning resources.")
        lines.append("")
    else:
        lines.append("OVERNIGHT VOLUME: Within normal range. No spikes detected.")
        lines.append("")

    if queue:
        top           = queue[0]
        cluster_count = sum(1 for i in queue if i.get("cluster_note"))
        lines.append(
            f"BOTTOM LINE: Start with {top['complaint']} in {top['location']} — "
            f"highest priority in your queue and {top['overdue']}."
        )
        if cluster_count:
            lines.append(f"{cluster_count} cluster(s) identified — one dispatch may cover multiple open cases.")
    else:
        lines.append("BOTTOM LINE: No action required this morning.")

    lines.append("")
    lines.append("—")
    lines.append("AirClaw Autonomous Pipeline | NYC 311 Operations Intelligence")
    lines.append("Generated automatically at 6am daily. Reply to confirm receipt.")

    briefing_text = "\n".join(lines)

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"Briefing drafted for {input.supervisor} ({input.agency}). "
            f"{len(queue)} prioritized action item(s). "
            f"{len(spikes)} spike alert(s). Ready to send."
        ),
        data={
            "agency":       input.agency,
            "supervisor":   input.supervisor,
            "briefing":     briefing_text,
            "action_count": len(queue),
            "spike_count":  len(spikes),
        },
        tool="draft_supervisor_briefing"
    )


def generate_summary(input: GenerateSummaryInput) -> AgentResult:
    lines = [f"AirClaw morning run complete — {datetime.now().strftime('%Y-%m-%d %H:%M')}."]
    if input.overnight_total:
        lines.append(f"Processed {input.overnight_total} overnight 311 requests.")
    if input.all_breaches:
        agencies_hit = list({b.get("agency", "") for b in input.all_breaches})
        lines.append(f"{len(input.all_breaches)} SLA breach(es) identified across {len(agencies_hit)} agency/agencies: {', '.join(agencies_hit)}.")
    else:
        lines.append("No SLA breaches. All open requests within response windows.")
    if input.all_spikes:
        top = input.all_spikes[0]
        lines.append(f"{len(input.all_spikes)} complaint spike(s) detected. Largest: {top.get('complaint_type', '')} up {top.get('increase_pct', '')} overnight.")
    if input.briefings:
        lines.append(f"{len(input.briefings)} supervisor briefing(s) drafted and ready to send: {', '.join(input.briefings)}.")
    lines.append("No manual triage required. Supervisors have been briefed. Pipeline standing by.")
    summary = " ".join(lines)
    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=summary,
        data={"summary": summary, "total_breaches": len(input.all_breaches), "total_spikes": len(input.all_spikes), "briefings_sent": input.briefings},
        tool="generate_summary"
    )


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "validate_schema":           validate_schema,
    "check_sla_breaches":        check_sla_breaches,
    "detect_complaint_spike":    detect_complaint_spike,
    "draft_supervisor_briefing": draft_supervisor_briefing,
    "generate_summary":          generate_summary,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "validate_schema",
            "description": "Verify the upstream 311 CSV has all required fields. Always call this first. Returns ESCALATE with exact diagnosis if schema has drifted.",
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
            "name": "check_sla_breaches",
            "description": "Find all open 311 requests past their SLA window. Returns specific case IDs grouped by agency. Call after validate_schema. Then call prioritize_queue on each agency's cases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":     {"type": "string"},
                    "agency_filter": {"type": "string", "description": "Filter to one agency e.g. NYPD. Empty = all agencies."},
                    "hours_window":  {"type": "integer", "description": "Only surface requests from last N hours (default 24)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_complaint_spike",
            "description": "Compare overnight complaint volume to 7-day rolling baseline. Flags complaint types that spiked significantly. Call after check_sla_breaches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":       {"type": "string"},
                    "spike_threshold": {"type": "number", "description": "Fractional increase to flag (default 0.4 = 40%)"},
                    "hours_window":    {"type": "integer", "description": "Overnight window to compare (default 24 hours)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_supervisor_briefing",
            "description": (
                "Write a ready-to-send morning briefing for ONE agency supervisor. "
                "Reads breach data directly from file_path, prioritizes internally, "
                "and produces a ranked action list with dispatch clustering. "
                "Call once per agency that has SLA breaches. "
                "Required fields: agency (string e.g. \'NYPD\'), "
                "supervisor (string e.g. \'Lt. Marcus Webb\'), "
                "file_path (string — same file path used in previous tool calls)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agency":     {"type": "string", "description": "Agency code exactly as returned by check_sla_breaches e.g. NYPD"},
                    "supervisor": {"type": "string", "description": "Supervisor name exactly as returned by check_sla_breaches e.g. Lt. Marcus Webb"},
                    "file_path":  {"type": "string", "description": "Same file path used in validate_schema and check_sla_breaches"},
                    "spikes":     {"type": "array", "items": {"type": "object"}, "description": "Optional spike records from detect_complaint_spike"},
                    "as_of":      {"type": "string", "description": "Optional timestamp string for the briefing header"}
                },
                "required": ["agency", "supervisor", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary",
            "description": "Produce a one-paragraph duty manager overview of everything the agent found and actioned. Call last — this is the final XCom output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "all_breaches":    {"type": "array", "items": {"type": "object"}},
                    "all_spikes":      {"type": "array", "items": {"type": "object"}},
                    "briefings":       {"type": "array", "items": {"type": "string"}},
                    "overnight_total": {"type": "integer"},
                    "original_goal":   {"type": "string"}
                },
                "required": []
            }
        }
    }
]