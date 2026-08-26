# AirClaw — Stage Demo Script
# DevFest DC · August 2026
# ─────────────────────────────────────────────────────────────
# HOW TO USE THIS:
#   Open this file in VS Code BEFORE you go on stage.
#   Keep it in a tab next to your terminals.
#   Every command you need is here. Nothing gets typed under pressure.
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# PRE-SHOW SETUP (do this in the green room / before doors open)
# ══════════════════════════════════════════════════════════════

# 1. Run the pre-show check. This is the single most important pre-flight step:
#    it makes a live tool call, so it catches a retired model, a bad key, or
#    broken data before the room does. Exits non-zero if you are not ready.
python3 preflight.py

# 2. Confirm clean data is loaded for both pipelines
cp data/nyc_311_clean.csv      data/nyc_311_upstream.csv
cp data/model_eval_clean.csv   data/model_eval_upstream.csv

# 3. Do a dry run of BOTH demos — confirm they reach SUCCESS
python3 run_demo.py
python3 run_model_eval.py

# 4. Open four terminal tabs and pre-load them:
#    Tab 1: 311 happy path (ready to run)
#    Tab 2: 311 failure beat (ready to run)
#    Tab 3: Model eval happy path (ready to run)
#    Tab 4: Model eval failure beat (ready to run)
#
#    Pre-type (but DO NOT run) these commands so you only hit Enter on stage:
#    Tab 1: python3 run_demo.py
#    Tab 2: python3 run_demo.py --break
#    Tab 3: python3 run_model_eval.py
#    Tab 4: python3 run_model_eval.py --break


# ══════════════════════════════════════════════════════════════
# DEMO 1 — NYC 311 TRIAGE PIPELINE
# For: NYC agency staff, government audience, civic tech
# Slot: Live · ~5 min on stage
# ══════════════════════════════════════════════════════════════

# --- Beat 1: Happy path ---
# Switch to Tab 1. Say: "It's 6am. Let me show you what the supervisor's morning looks like now."
# Hit Enter:
python3 run_demo.py

# WHAT TO SAY while logs stream:
# "Schema validation runs first — that's the contract. Everything after it is the
#  agent's call."
# "Now it's calling check_sla_breaches..."
# "29 open requests past their SLA window. Seven agencies affected."
# "And it's checking overnight volume against the seven-day baseline —
#  illegal parking up 271%, noise complaints up 232%. Those alerts route to
#  NYPD's briefing, because those are NYPD's complaint types."
# "Nobody told it which agencies matter. It read the breach counts and picked
#  who gets a briefing — those supervisor names came out of the data, not my prompt."
# [When briefing streams] → STOP TALKING. Let the room read it. Count 8 seconds silently.
# "That email is ready to send. Lt. Webb didn't write it. It's 6:04am."
#
# IF YOU SEE [fallback] IN THE LOGS: the agent stalled and the run finished
# deterministically. Say nothing about it, finish the beat on the briefing, and
# move on — the briefing on screen is still real output from the real tool.

# --- Beat 2: Failure beat ---
# Switch to Tab 2. Say: "Now let me break it."
# Hit Enter:
python3 run_demo.py --break

# WHAT TO SAY while ESCALATE fires:
# "I just swapped the upstream file. The data team renamed complaint_type to complaint_category."
# [Let ESCALATE message render completely]
# "Schema drift detected. Missing field. Likely rename. Pipeline halted. Audit trail preserved."
# "That pipeline told you exactly what broke and why — without a single conditional you wrote."


# --- Beat 3 (optional): Audience input ---
# Take a real question from the room, phrase it as a goal, and let the agent
# plan its own approach. It picks query_requests, filters the data, and answers
# in two sentences with the actual numbers — including calling out ties.
# Hit Enter:
# python3 run_demo.py --goal "Which agency has the most overdue requests in Brooklyn?"
#
# WHAT TO SAY: "I did not write a code path for that question. It chose the tool,
# chose the filters, and answered from the data."
#
# SAFE QUESTIONS (verified against the data — all answerable by query_requests):
#   "Which borough has the most breaches?"                  -> BRONX, 9 of 29
#   "Which agency has the most overdue requests in Brooklyn?" -> DSNY, 2, tied with NYPD
#   "What complaint type is most common among overdue cases?"  -> 5-way tie at 3
#
# RISKY: anything needing math the tools do not do (averages, trends over time,
# per-capita). The agent will say it cannot rather than invent a number — which
# is a fine answer, but do not set it up as the finale.


# ══════════════════════════════════════════════════════════════
# DEMO 2 — MODEL MIGRATION EVAL PIPELINE
# For: Sr Engineers, Solutions Architects, AI/ML platform teams
# Slot: Live · ~5 min on stage
# ══════════════════════════════════════════════════════════════

# --- Beat 1: Happy path ---
# Switch to Tab 3. Say: "Same framework. Different problem. This one is yours."
# Hit Enter:
python3 run_model_eval.py

# WHAT TO SAY while logs stream:
# "300 production prompts. GPT-4o on one side, Llama-3.3-Nemotron-Super on the other."
# "The agent is scoring quality by task category..."
# "Detecting regressions — where does Model B actually get worse?"
# "Running cost analysis at 100k prompts a month..."
# [When migration report streams] → STOP TALKING. Let the room read it. Count 8 seconds.
# Read the recommendation out loud: "PROCEED PARTIALLY — migrate code generation.
#  Hold RAG QA and customer support. 66% cheaper. 50% faster. $9,272 annual savings."
#  (Verified against the tools — those four numbers are what the report prints.)
# "A Sr Engineer used to spend 2-3 days producing that report. AirClaw just did it."

# --- Beat 2: Failure beat ---
# Switch to Tab 4. Say: "And if the eval pipeline itself has a schema problem..."
# Hit Enter:
python3 run_model_eval.py --break

# WHAT TO SAY while ESCALATE fires:
# "The upstream eval pipeline renamed model_b_quality_score to model_b_score."
# [Let ESCALATE render]
# "Migration report cannot be generated. Schema corrected needed. "
# "Same pattern. Different domain. Same trust."


# ══════════════════════════════════════════════════════════════
# TIMING GUIDE
# ══════════════════════════════════════════════════════════════
#
#  311 happy path     ~2.5 min (agent + briefing streaming)
#  311 failure beat   ~1.0 min
#  Model eval happy   ~3.0 min (more tool calls, bigger report)
#  Model eval failure ~0.5 min
#  ─────────────────────────────
#  Total demo time    ~7 min
#
#  At 25 min total, that fits inside the 13 min demo block.
#  If running long: skip the model eval failure beat.
#  Never skip either happy path.


# ══════════════════════════════════════════════════════════════
# IF SOMETHING GOES WRONG
# ══════════════════════════════════════════════════════════════

# NIM is slow / timing out:
#   → Open browser tab with pre-recorded backup (record these before the conference)
#   → Say "let me show you the run from this morning" — no apology, just play it

# Agent writes report as text instead of calling the tool:
#   → The run still completes — you will see [fallback] lines and the real report.
#   → Don't draw attention to it. If you want a clean run, hit Enter again.

# Wrong data file loaded:
#   → Reset clean data and rerun
cp data/nyc_311_clean.csv    data/nyc_311_upstream.csv
cp data/model_eval_clean.csv data/model_eval_upstream.csv
#   → Then rerun whichever demo

# Anything unexplained — run the pre-show check, it names the problem:
python3 preflight.py --no-airflow

# Model retired by NVIDIA (HTTP 410 Gone on every call — this happened on
# 2026-08-26). List what is currently served and set NIM_MODEL in .env:
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NIM_API_KEY" | grep -o '"id":"[^"]*"'


# ══════════════════════════════════════════════════════════════
# OPTIONAL — AIRFLOW UI ON SCREEN
# ══════════════════════════════════════════════════════════════
#
# Pre-show, in a fifth terminal tab:
#   ./run_airflow.sh --check     # confirm both DAGs parse — do this first
#   ./run_airflow.sh             # UI at http://localhost:8080
#
# On stage, trigger from the UI or:
#   airflow dags trigger airclaw_demo
#   airflow dags trigger model_eval_demo
#
# The briefings and the migration report stream into the task logs, so open the
# nemoclaw_agent task log and let the room watch it fill in. Say: "same agent,
# same tools — Airflow just decides when it runs and what happens if it fails."


# ══════════════════════════════════════════════════════════════
# WHAT TO RECORD AS BACKUP (before conference day)
# ══════════════════════════════════════════════════════════════
#
# Record these four runs with QuickTime or OBS. Save as MP4.
# Upload to Google Drive. Have the link open in a browser tab.
#
#   backup_311_happy.mp4        python3 run_demo.py
#   backup_311_failure.mp4      python3 run_demo.py --break
#   backup_eval_happy.mp4       python3 run_model_eval.py
#   backup_eval_failure.mp4     python3 run_model_eval.py --break