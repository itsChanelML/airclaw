# AirClaw — Stage Demo Script
# AI DevSummit NYC · June 9–10, 2026
# ─────────────────────────────────────────────────────────────
# HOW TO USE THIS:
#   Open this file in VS Code BEFORE you go on stage.
#   Keep it in a tab next to your terminals.
#   Every command you need is here. Nothing gets typed under pressure.
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# PRE-SHOW SETUP (do this in the green room / before doors open)
# ══════════════════════════════════════════════════════════════

# 1. Set your NIM key (if not already in .env)
export NIM_API_KEY=your_key_here

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
# "Watch the logs. The agent is validating schema first — it decided to do that, I didn't script it."
# "Now it's calling check_sla_breaches..."
# "29 open requests past their SLA window. Seven agencies affected."
# [When briefing streams] → STOP TALKING. Let the room read it. Count 8 seconds silently.
# "That email is ready to send. Lt. Webb didn't write it. It's 6:04am."

# --- Beat 2: Failure beat ---
# Switch to Tab 2. Say: "Now let me break it."
# Hit Enter:
python3 run_demo.py --break

# WHAT TO SAY while ESCALATE fires:
# "I just swapped the upstream file. The data team renamed complaint_type to complaint_category."
# [Let ESCALATE message render completely]
# "Schema drift detected. Missing field. Likely rename. Pipeline halted. Audit trail preserved."
# "That pipeline told you exactly what broke and why — without a single conditional you wrote."


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
# Read the recommendation out loud: "PROCEED PARTIALLY — migrate code generation and RAG.
#  Hold customer support. 66% cheaper. 50% faster. $9,272 annual savings."
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

# Agent writes report as text instead of calling tool:
#   → This is a known failure mode — just re-run
#   → python3 run_model_eval.py   (Tab 3, hit Enter again)

# Wrong data file loaded:
#   → Reset clean data and rerun
cp data/nyc_311_clean.csv    data/nyc_311_upstream.csv
cp data/model_eval_clean.csv data/model_eval_upstream.csv
#   → Then rerun whichever demo

# NIM_API_KEY not set:
export NIM_API_KEY=your_key_here


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