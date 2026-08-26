"""
Date rebasing
-------------
Shifts the demo data forward so it always looks like it was collected in the
last few days, no matter when the demo runs.

Why this exists
---------------
The 311 sample was generated in June 2024 and its timestamps were frozen there,
while `detect_complaint_spike` and the DAG's ingest task both window off
`datetime.now()`. Run the demo in 2026 and you get:

  - "No complaint spikes detected" on every run — the overnight and baseline
    windows are both empty, so the spike capability never demonstrates itself
  - "[ingest] Overnight (24h) : 0" in the Airflow log, one task before the agent
    reports "Processed 300 overnight 311 requests"
  - case IDs reading SR-2024-… under a briefing dated today

How it works
------------
The sample is internally consistent: for every open request,
`hours_open == (newest_created_date - created_date)`, verified to within 0.08h.
So a single uniform shift — mapping the newest record to "now" — restores every
relationship at once, and leaves `hours_open`, `sla_hours` and `sla_breach`
untouched. Breach counts, agency rankings and supervisor assignments come out
identical, which matters because those numbers are in the talk track.

Case ID year prefixes are rewritten to match their shifted dates.

Python 3.9 compatible.
"""

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Columns holding an ISO-8601 timestamp that should move with the shift.
DATE_COLUMNS = ("created_date", "closed_date")

# SR-2024-20000079 -> the year segment is rewritten to match the shifted date.
CASE_ID_YEAR = re.compile(r"^([A-Z]+)-(\d{4})-(\d+)$")

ISO = "%Y-%m-%dT%H:%M:%S"


def _parse(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:19], ISO)
    except ValueError:
        return None


def rebase_rows(rows: List[dict], now: Optional[datetime] = None) -> timedelta:
    """
    Shift every timestamp in `rows` so the newest created_date lands on `now`.
    Mutates rows in place and returns the shift that was applied.
    """
    now = now or datetime.now()

    newest = None
    for row in rows:
        dt = _parse(row.get("created_date", ""))
        if dt and (newest is None or dt > newest):
            newest = dt

    if newest is None:
        return timedelta(0)

    shift = now - newest

    for row in rows:
        for column in DATE_COLUMNS:
            dt = _parse(row.get(column, ""))
            if dt:
                row[column] = (dt + shift).strftime(ISO)

        # Keep the case ID's year in step with its (shifted) creation date.
        created = _parse(row.get("created_date", ""))
        match   = CASE_ID_YEAR.match(row.get("unique_key", "") or "")
        if created and match:
            prefix, _old_year, serial = match.groups()
            row["unique_key"] = f"{prefix}-{created.year}-{serial}"

    return shift


def _snapshot(rows: List[dict]) -> dict:
    """Facts that must survive rebasing — these numbers are in the talk track."""
    from collections import Counter
    breaches = [r for r in rows if r.get("sla_breach", "").strip() == "YES"]
    return {
        "total":       len(rows),
        "breaches":    len(breaches),
        "by_agency":   dict(Counter(r.get("agency", "") for r in breaches)),
        "supervisors": dict(Counter(r.get("supervisor", "") for r in breaches)),
        "durations":   sorted(
            round(d, 1) for d in
            (_duration_hours(r) for r in rows) if d is not None
        ),
    }


def _verify(before: dict, after: dict, rows: List[dict], now: datetime) -> None:
    """Raise if rebasing changed anything the demo narrative depends on."""
    for key in ("total", "breaches", "by_agency", "supervisors", "durations"):
        if before[key] != after[key]:
            raise AssertionError(
                f"Rebasing changed {key}: {before[key]!r} -> {after[key]!r}. "
                f"Refusing to write data that contradicts the demo."
            )

    for row in rows:
        created = _parse(row.get("created_date", ""))
        closed  = _parse(row.get("closed_date", ""))
        if created and created > now:
            raise AssertionError(f"{row.get('unique_key')} created in the future.")
        if closed and closed > now:
            raise AssertionError(f"{row.get('unique_key')} closed in the future.")

        # An open request's hours_open must still match its creation date, and a
        # non-breach must not have quietly aged past its SLA.
        if row.get("status", "") != "Closed" and row.get("hours_open", "") and created:
            age = (now - created).total_seconds() / 3600
            if abs(float(row["hours_open"]) - age) > 0.5:
                raise AssertionError(
                    f"{row.get('unique_key')} hours_open={row['hours_open']} "
                    f"disagrees with its created_date (age {age:.1f}h)."
                )
            try:
                sla = float(row.get("sla_hours", "") or 0)
            except ValueError:
                sla = 0
            breached = row.get("sla_breach", "").strip() == "YES"
            if sla and not breached and age > sla:
                raise AssertionError(
                    f"{row.get('unique_key')} is marked NO breach but is now "
                    f"{age:.1f}h old against a {sla:.0f}h SLA."
                )


def rebase_csv(src, dst, now: Optional[datetime] = None,
               reshape: bool = True) -> timedelta:
    """
    Read `src`, shift its dates to `now`, write the result to `dst`.

    Replaces the plain `shutil.copy` the demo runners used to do. Safe to call
    repeatedly — it always rebases from the pristine source file. Verifies the
    demo's load-bearing numbers survived, and raises if they did not.
    """
    src, dst = Path(src), Path(dst)
    now = now or datetime.now()

    with open(src, newline="") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows       = list(reader)

    before = _snapshot(rows)
    shift  = rebase_rows(rows, now)

    # Only the 311 feed has an SLA/status model to reshape; the eval file has no
    # timestamps at all and passes straight through.
    if reshape and rows and "sla_breach" in rows[0] and "status" in rows[0]:
        reshape_311(rows, now)

    _verify(before, _snapshot(rows), rows, now)

    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return shift


def describe_shift(shift: timedelta) -> str:
    """Human-readable summary for the demo's setup line."""
    days = shift.days
    if abs(days) < 1:
        hours = round(shift.total_seconds() / 3600, 1)
        return f"{hours:+} hours"
    return f"{days:+,} days"


if __name__ == "__main__":
    # Handy for eyeballing the result: python3 rebase_data.py <src> <dst>
    import sys
    if len(sys.argv) != 3:
        print("usage: python3 rebase_data.py <source.csv> <dest.csv>")
        raise SystemExit(2)
    applied = rebase_csv(sys.argv[1], sys.argv[2])
    print(f"Rebased {sys.argv[1]} -> {sys.argv[2]} ({describe_shift(applied)})")


# ── Distribution reshaping ─────────────────────────────────────────────────────
# The uniform shift alone leaves the sample's original shape: 183 of 300
# requests created in the final 24 hours, against 117 spread over the previous
# week. That starves the spike baseline, so detect_complaint_spike reports 20
# spiking complaint types with increases like "4800%" — which reads as a broken
# baseline rather than a surge, and buries the real signal.
#
# This spreads the movable records across the baseline window so overnight
# volume is comparable to the daily average, leaving a deliberate, believable
# overnight cluster in the noise-complaint types (a real NYC summer pattern).
#
# What it must never change — these numbers are in the talk track:
#   - the 29 SLA breaches, their agencies, supervisors and hours overdue
#   - closed requests' resolution durations
#   - hours_open's agreement with created_date
# Breach rows are therefore never moved, and every other row carries bounds:
#   open non-breach : age must stay under sla_hours, or it would become a breach
#   closed          : age must exceed its own resolution time, or it would have
#                     been closed in the future
# Invariants are asserted at the end; a violation raises rather than quietly
# shipping data that contradicts the talk.

BASELINE_DAYS  = 7
SPAN_HOURS     = BASELINE_DAYS * 24
# Records may be placed one day beyond the baseline window: a request that took
# 184 hours to close cannot be less than 184 hours old without having been
# closed in the future. The original sample spans 7d22h for the same reason.
PLACEMENT_DAYS = BASELINE_DAYS + 1
# Complaint types deliberately clustered overnight — the spike the demo shows.
SPIKE_TYPES    = ("Noise - Residential", "Noise - Street/Sidewalk", "Illegal Parking")
SPIKE_BOOST    = 2.6
OVERNIGHT_SIZE = 48


def _duration_hours(row) -> Optional[float]:
    created = _parse(row.get("created_date", ""))
    closed  = _parse(row.get("closed_date", ""))
    if created and closed:
        return (closed - created).total_seconds() / 3600
    return None


def _set_age(row, now: datetime, age_hours: float) -> None:
    """Place a row `age_hours` before `now`, preserving its resolution duration."""
    duration = _duration_hours(row)
    created  = now - timedelta(hours=age_hours)
    row["created_date"] = created.strftime(ISO)

    if duration is not None:
        row["closed_date"] = (created + timedelta(hours=duration)).strftime(ISO)

    # Open requests advertise how long they have been open; keep that true.
    if row.get("status", "") != "Closed" and row.get("hours_open", ""):
        row["hours_open"] = str(round(age_hours, 1))

    match = CASE_ID_YEAR.match(row.get("unique_key", "") or "")
    if match:
        prefix, _year, serial = match.groups()
        row["unique_key"] = f"{prefix}-{created.year}-{serial}"


def reshape_311(rows: List[dict], now: datetime) -> dict:
    """
    Redistribute non-breach requests so overnight volume reads as a plausible
    surge rather than the whole file. Returns a summary dict for logging.
    """
    complaint_key = "complaint_type" if "complaint_type" in (rows[0] if rows else {}) else None

    breaches, movable = [], []
    for row in rows:
        if row.get("sla_breach", "").strip() == "YES":
            breaches.append(row)
            continue

        duration = _duration_hours(row)
        try:
            sla = float(row.get("sla_hours", "") or SPAN_HOURS)
        except ValueError:
            sla = SPAN_HOURS

        is_closed = row.get("status", "") == "Closed"
        # Half-hour margins keep rounding from pushing a row over a boundary.
        min_age = (duration + 0.5) if (is_closed and duration is not None) else 0.5
        # A closed request must be at least as old as it took to resolve.
        max_age = max(SPAN_HOURS + 24, min_age + 1.0) if is_closed else (sla - 0.5)
        movable.append({"row": row, "min_age": min_age, "max_age": max_age})

    if not movable:
        return {"overnight": 0, "moved": 0, "breaches": len(breaches)}

    # ── Choose the overnight set ───────────────────────────────────────────────
    # Anything that cannot legally be older than a day is overnight by force.
    forced     = [m for m in movable if m["max_age"] <= 24]
    forced_ids = {id(m) for m in forced}
    eligible   = [m for m in movable
                  if m["min_age"] <= 23 and id(m) not in forced_ids]

    shares = collections_counter(
        (m["row"].get(complaint_key, "") if complaint_key else "") for m in movable
    )
    total  = sum(shares.values()) or 1

    overnight = list(forced)
    have      = collections_counter(
        (m["row"].get(complaint_key, "") if complaint_key else "") for m in overnight
    )

    # Fill toward a proportional overnight mix, boosting the spike types so the
    # surge is concentrated where it is plausible instead of spread everywhere.
    def quota(ctype: str) -> int:
        base = OVERNIGHT_SIZE * (shares[ctype] / total)
        if ctype in SPIKE_TYPES:
            base *= SPIKE_BOOST
        return int(round(base))

    # Deterministic ordering — the demo must reproduce run to run.
    eligible.sort(key=lambda m: (m["row"].get("unique_key", ""),))
    for m in eligible:
        if len(overnight) >= OVERNIGHT_SIZE:
            break
        ctype = m["row"].get(complaint_key, "") if complaint_key else ""
        if have[ctype] < quota(ctype):
            overnight.append(m)
            have[ctype] += 1

    overnight_ids = {id(m) for m in overnight}
    baseline      = [m for m in movable if id(m) not in overnight_ids]

    # ── Place the overnight set inside the last 24 hours ───────────────────────
    for index, m in enumerate(overnight):
        lo   = max(m["min_age"], 0.5)
        hi   = min(m["max_age"], 23.5)
        span = max(hi - lo, 0.1)
        # Spread across the night rather than stacking on one timestamp.
        _set_age(m["row"], now, lo + span * ((index % 12) / 12.0))

    # ── Spread the rest across the previous seven days ────────────────────────
    # Longest-constrained rows first, so they get the deep days they need.
    # Day d covers ages [d*24, (d+1)*24) — so day 1 begins at 24h, OUTSIDE the
    # overnight window. Getting this boundary wrong silently refills the
    # overnight window with "baseline" rows and starves the baseline again.
    baseline.sort(key=lambda m: (-m["min_age"], m["row"].get("unique_key", "")))
    day_counts = {day: 0 for day in range(1, PLACEMENT_DAYS + 1)}
    for m in baseline:
        feasible = [
            day for day in range(1, PLACEMENT_DAYS + 1)
            if m["min_age"] <= (day + 1) * 24 - 0.5 and m["max_age"] >= day * 24 + 0.5
        ]
        if not feasible:
            # Bounds cannot be satisfied in any day bucket. min_age is always a
            # legal age by construction, so use it rather than corrupt the row.
            _set_age(m["row"], now, m["min_age"] + 0.5)
            continue
        # Fill the emptiest feasible day to even out the week.
        day = min(feasible, key=lambda d: (day_counts[d], d))
        day_counts[day] += 1
        lo = max(m["min_age"], day * 24 + 0.5)
        hi = min(m["max_age"], (day + 1) * 24 - 0.5)
        _set_age(m["row"], now, lo + max(hi - lo, 0.0) * ((day_counts[day] % 10) / 10.0))

    return {
        "overnight": len(overnight),
        "baseline":  len(baseline),
        "breaches":  len(breaches),
        "per_day":   day_counts,
    }


def collections_counter(iterable):
    from collections import Counter
    return Counter(iterable)
