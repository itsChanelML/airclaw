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

from schema_diff import suggest_renames


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
    file_path:            str
    baseline_days:        int   = 7
    spike_threshold:      float = 0.40
    hours_window:         int   = 24
    # Ignore types with fewer than this many overnight requests — percentage
    # jumps off a near-zero baseline are noise, not signal.
    min_overnight_volume: int   = 5


class PrioritizeQueueInput(BaseModel):
    agency:   str
    breaches: List[Dict[str, Any]]
    spikes:   List[Dict[str, Any]] = []


class QueryRequestsInput(BaseModel):
    """Ad-hoc analytical question over the 311 feed."""
    file_path:      str
    group_by:       str  = "complaint_type"   # complaint_type|borough|district|agency|status|supervisor
    only_breaches:  bool = False              # restrict to SLA-breached requests
    only_open:      bool = False              # exclude Closed requests
    borough:        str  = ""                 # optional filters
    agency:         str  = ""
    complaint_type: str  = ""
    top_n:          int  = 5


class DraftBriefingInput(BaseModel):
    agency:      str
    supervisor:  str
    file_path:   str
    spikes:      List[Dict[str, Any]] = []
    as_of:       str = ""


class GenerateSummaryInput(BaseModel):
    file_path:       str = ""
    # Optional overrides. Normally left empty — the tool recomputes from
    # file_path rather than requiring the agent to echo every breach record
    # back through message history.
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
        suggestions = suggest_renames(input.required_fields, list(actual_fields), missing)
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

        # A rare complaint type going from 0.3/day to 4 overnight is a 1300%
        # increase and operationally meaningless. Requiring a floor on absolute
        # volume keeps the small-denominator noise out of the briefing so the
        # real cluster is what a supervisor sees.
        if overnight_vol < input.min_overnight_volume:
            continue

        if increase >= input.spike_threshold:
            spikes.append({
                "complaint_type":  ct,
                "overnight_count": overnight_vol,
                "baseline_avg":    round(avg_daily, 1),
                "increase_pct":    f"{round(increase * 100)}%",
                "severity":        "HIGH" if increase >= 1.0 else "MODERATE",
            })

    # Rank by absolute excess volume — 13 requests against a 2.3/day baseline
    # matters more than 4 against 0.3/day, even though the percentage is lower.
    spikes.sort(key=lambda x: x["overnight_count"] - x["baseline_avg"], reverse=True)

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


def query_requests(input: QueryRequestsInput) -> AgentResult:
    """
    Answer a counting question about the 311 feed: group requests by a field,
    optionally filtered, and return the ranked counts.

    Exists so the agent can answer questions the fixed pipeline tools do not
    cover — "which complaint type is most common among the overdue cases",
    "which borough has the most breaches". Without it, an off-script question
    leaves the agent with no way to reach the data at all.
    """
    path = Path(input.file_path)
    if not path.exists():
        return AgentResult(
            status=AgentStatus.ESCALATE,
            message=f"File not found: {input.file_path}",
            tool="query_requests"
        )

    VALID = ("complaint_type", "borough", "district", "agency", "status", "supervisor")
    group_by = (input.group_by or "").strip()
    if group_by not in VALID:
        return AgentResult(
            status=AgentStatus.RETRY,
            message=f"group_by must be one of {list(VALID)}, got '{input.group_by}'.",
            tool="query_requests"
        )

    counts  = defaultdict(int)
    matched = 0
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if input.only_breaches and row.get("sla_breach", "").strip() != "YES":
                    continue
                if input.only_open and row.get("status", "").strip() == "Closed":
                    continue
                if input.borough and row.get("borough", "") != input.borough:
                    continue
                if input.agency and row.get("agency", "") != input.agency:
                    continue
                if input.complaint_type and row.get("complaint_type", "") != input.complaint_type:
                    continue
                counts[row.get(group_by, "") or "UNKNOWN"] += 1
                matched += 1
    except Exception as e:
        return AgentResult(
            status=AgentStatus.RETRY,
            message=f"Error reading data: {e}",
            tool="query_requests"
        )

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(input.top_n, 1)]

    scope = []
    if input.only_breaches: scope.append("SLA-breached")
    if input.only_open:     scope.append("open")
    if input.borough:       scope.append(f"in {input.borough}")
    if input.agency:        scope.append(f"for {input.agency}")
    if input.complaint_type: scope.append(f"of type {input.complaint_type}")
    scope_text = " ".join(scope) or "all"

    if not ranked:
        return AgentResult(
            status=AgentStatus.SUCCESS,
            message=f"No {scope_text} requests matched that query.",
            data={"group_by": group_by, "results": [], "total_matched": 0},
            tool="query_requests"
        )

    top_label, top_count = ranked[0]
    breakdown = ", ".join(f"{label}: {count}" for label, count in ranked)

    # Call out ties explicitly. "DSNY has the most with 2" is misleading when
    # NYPD also has 2, and that is exactly the kind of detail an audience
    # catches. Ties are counted across the whole result set, not just top_n.
    tied = sorted(label for label, count in counts.items() if count == top_count)
    if len(tied) > 1:
        others   = [t for t in tied if t != top_label]
        top_text = (
            f"Highest: {top_count} each, tied between {', '.join(tied)}"
            if len(tied) > 2 else
            f"Highest: {top_label} with {top_count}, tied with {others[0]}"
        )
    else:
        top_text = f"Highest: {top_label} with {top_count}"

    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=(
            f"{matched} {scope_text} request(s) grouped by {group_by}. "
            f"{top_text}. "
            f"Top {len(ranked)} — {breakdown}."
        ),
        data={
            "group_by":      group_by,
            "total_matched": matched,
            "top":           {"label": top_label, "count": top_count, "tied_with": tied[1:]},
            "results":       [{"label": l, "count": c} for l, c in ranked],
        },
        tool="query_requests"
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

    # ── Self-hydration ────────────────────────────────────────────────────────
    # The agent passes spikes=[] because the spike payload is stripped from
    # message history, which used to print "No spikes detected" in the briefing
    # while the pipeline had just reported three. Recompute here, and keep only
    # the complaint types this agency actually handles — DSNY does not need an
    # alert about noise complaints.
    if not spikes:
        detected = detect_complaint_spike(SpikeDetectionInput(file_path=input.file_path))
        if detected.status == AgentStatus.SUCCESS:
            agency_types = set()
            try:
                with open(Path(input.file_path), newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("agency", "") == input.agency:
                            agency_types.add(row.get("complaint_type", ""))
            except Exception:
                agency_types = set()
            spikes = [
                sp for sp in detected.data.get("spikes", [])
                if not agency_types or sp.get("complaint_type", "") in agency_types
            ]

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
    breaches = input.all_breaches
    spikes   = input.all_spikes
    total    = input.overnight_total

    # ── Self-hydration ────────────────────────────────────────────────────────
    # Without this, an agent that (correctly) does not echo the full breach list
    # back gets a summary reading "No SLA breaches" directly after briefings for
    # 29 of them streamed past. The duty manager summary has to agree with the
    # briefings, so recompute from the source instead of trusting the arguments.
    if input.file_path:
        if not breaches:
            r = check_sla_breaches(SLABreachInput(file_path=input.file_path))
            if r.status == AgentStatus.SUCCESS:
                breaches = r.data.get("breaches", [])
        if not spikes:
            r = detect_complaint_spike(SpikeDetectionInput(file_path=input.file_path))
            if r.status == AgentStatus.SUCCESS:
                spikes = r.data.get("spikes", [])

    # The agent supplies overnight_total by hand and tends to pass the whole
    # file size. Count what actually arrived overnight so the summary cannot
    # claim 300 overnight requests when 48 came in.
    overnight_count = None
    feed_total      = None
    if input.file_path:
        try:
            cutoff = datetime.now() - timedelta(hours=24)
            with open(Path(input.file_path), newline="") as f:
                rows_read = list(csv.DictReader(f))
            feed_total      = len(rows_read)
            overnight_count = sum(
                1 for r in rows_read
                if (r.get("created_date", "")[:19] or "") >= cutoff.strftime("%Y-%m-%dT%H:%M:%S")
            )
        except Exception:
            overnight_count = None

    lines = [f"AirClaw morning run complete — {datetime.now().strftime('%Y-%m-%d %H:%M')}."]
    if overnight_count is not None:
        lines.append(
            f"Processed {overnight_count} request(s) received overnight "
            f"({feed_total} in the feed)."
        )
    elif total:
        lines.append(f"Processed {total} overnight 311 requests.")
    if breaches:
        agencies_hit = sorted({b.get("agency", "") for b in breaches if b.get("agency")})
        lines.append(
            f"{len(breaches)} SLA breach(es) identified across "
            f"{len(agencies_hit)} agency/agencies: {', '.join(agencies_hit)}."
        )
    else:
        lines.append("No SLA breaches. All open requests within response windows.")
    if spikes:
        top = spikes[0]
        lines.append(
            f"{len(spikes)} complaint spike(s) detected. Largest: "
            f"{top.get('complaint_type', '')} up {top.get('increase_pct', '')} overnight."
        )
    if input.briefings:
        lines.append(
            f"{len(input.briefings)} supervisor briefing(s) drafted and ready to send: "
            f"{', '.join(input.briefings)}."
        )
    lines.append("No manual triage required. Supervisors have been briefed. Pipeline standing by.")
    summary = " ".join(lines)
    return AgentResult(
        status=AgentStatus.SUCCESS,
        message=summary,
        data={
            "summary":        summary,
            "total_breaches": len(breaches),
            "total_spikes":   len(spikes),
            "briefings_sent": input.briefings,
        },
        tool="generate_summary"
    )


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "validate_schema":           validate_schema,
    "check_sla_breaches":        check_sla_breaches,
    "detect_complaint_spike":    detect_complaint_spike,
    "query_requests":            query_requests,
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
            "name": "query_requests",
            "description": (
                "Answer a counting question about the 311 feed. Groups requests by one "
                "field and returns ranked counts, with optional filters. Use this for any "
                "question the other tools do not directly answer — for example the most "
                "common complaint type among overdue cases (group_by='complaint_type', "
                "only_breaches=true), or which borough carries the most breaches "
                "(group_by='borough', only_breaches=true)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":      {"type": "string", "description": "Same file path used in the previous tool calls"},
                    "group_by":       {"type": "string", "enum": ["complaint_type", "borough", "district", "agency", "status", "supervisor"], "description": "Field to group and count by"},
                    "only_breaches":  {"type": "boolean", "description": "Restrict to requests that breached their SLA"},
                    "only_open":      {"type": "boolean", "description": "Exclude closed requests"},
                    "borough":        {"type": "string", "description": "Optional borough filter, e.g. BROOKLYN"},
                    "agency":         {"type": "string", "description": "Optional agency filter, e.g. NYPD"},
                    "complaint_type": {"type": "string", "description": "Optional complaint type filter"},
                    "top_n":          {"type": "integer", "description": "How many groups to return (default 5)"}
                },
                "required": ["file_path", "group_by"]
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
            "description": "Produce a one-paragraph duty manager overview of everything the agent found and actioned. Call last — this is the final XCom output. Pass file_path and the list of supervisors you briefed; breach and spike counts are recomputed from the file, so do not echo those payloads back.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path":       {"type": "string", "description": "Same file path used in the previous tool calls"},
                    "briefings":       {"type": "array", "items": {"type": "string"}, "description": "Names of the supervisors you drafted briefings for"},
                    "overnight_total": {"type": "integer", "description": "Total requests processed, e.g. 300"},
                    "original_goal":   {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    }
]

# ── History trimming ───────────────────────────────────────────────────────────
# Tool results carry full nested payloads (every breach record, every case).
# Feeding those back into message history burns the context window and makes
# the model lose the thread. This keeps status + message always, plus the
# by_agency summary the agent actually needs to pick its next call.
#
# Lives here rather than in the runner so run_demo.py and the Airflow
# NemoClawOperator trim identically — the operator used to skip trimming
# entirely, which is why the Airflow path degraded where the runner didn't.

import json as _json


def trim_for_history(tool_name: str, result_json: str) -> str:
    try:
        obj  = _json.loads(result_json)
        data = obj.get("data", {})

        if tool_name == "check_sla_breaches" and data:
            trimmed_by_agency = {}
            for agency, info in data.get("by_agency", {}).items():
                trimmed_by_agency[agency] = {
                    "count":      info.get("count", 0),
                    "supervisor": info.get("supervisor", ""),
                    "cases":      info.get("cases", [])[:3],
                }
            obj["data"] = {
                "by_agency": trimmed_by_agency,
                "total":     data.get("total", 0),
                "agencies":  data.get("agencies", []),
            }
        else:
            obj.pop("data", None)

        return _json.dumps(obj)
    except Exception:
        return result_json[:800]
