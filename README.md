# AirClaw 🦾
### Autonomous Productivity Agents with Apache Airflow + NemoClaw
**AI DevSummit New York · June 9–10, 2026**

> "What if your pipeline could think for itself?"

---

## What this is

AirClaw pairs **Apache Airflow** (orchestration) with **NemoClaw** — the [OpenClaw](https://github.com/openclawai/openclaw) agent framework running `nvidia/llama-3.3-nemotron-super-49b-v1` via NVIDIA NIM — to create autonomous pipelines that reason, adapt, and fail gracefully without human intervention.

Two pipelines. Same framework. Two different problems.

**Pipeline 1 — NYC 311 Triage Agent**
Every morning a city agency supervisor opens their laptop to 300 overnight 311 service requests. Some are overdue past their SLA window. Some are complaint spikes. Someone needs to find all of that, figure out who to call, and write the briefing. That someone is a person — and it takes them 30 to 60 minutes every single morning. AirClaw does it instead. The supervisor's first action of the day is approving work the pipeline already did.

**Pipeline 2 — Model Migration Eval Agent**
A team wants to migrate from GPT-4o to Llama-3.3-Nemotron-Super in production. Someone needs to run both models on production prompts, compare quality by task category, detect regressions, project cost savings, and write a go/no-go recommendation. That analysis takes a Sr Engineer 2-3 days manually. AirClaw does it in one pipeline run.

---

## Architecture

```
[Airflow DAG]  →  [NemoClawOperator]  →  [Tool Registry]
      ↕                   ↕                      ↕
  Schedule            Reasons + Acts         Python callables
  Contract            Calls tools            Typed schemas
  Monitor             Returns result         RETRY/ESCALATE/SUCCESS
```

**Three layers:**

- **Airflow** — defines what must happen and when. Doesn't care how.
- **NemoClaw** — OpenClaw + `nvidia/llama-3.3-nemotron-super-49b-v1` via NIM. Owns the how.
- **Tool Registry** — typed tools with `SUCCESS / RETRY / ESCALATE` contracts. No silent failures.

---

## Project structure

```
airclaw/
├── dags/
│   ├── airclaw_demo.py          # 311 triage DAG — 3 tasks, 6am schedule
│   └── model_eval_demo.py       # Model eval DAG — triggered on eval completion
├── data/
│   ├── nyc_311_clean.csv        # Happy path — 300 requests, 29 SLA breaches
│   ├── nyc_311_broken.csv       # Broken schema — complaint_type → complaint_category
│   ├── nyc_311_upstream.csv     # Active file — swapped by run_demo.py automatically
│   ├── model_eval_clean.csv     # Happy path — 300 prompts, GPT-4o vs Nemotron-Super
│   ├── model_eval_broken.csv    # Broken schema — model_b_quality_score → model_b_score
│   └── model_eval_upstream.csv  # Active file — swapped by run_model_eval.py automatically
├── plugins/
│   └── nemoclaw_operator.py     # Custom Airflow operator + plain-English log formatter
├── tools/
│   ├── airclaw_tools.py         # 311 tool registry — 6 tools, Pydantic schemas
│   └── model_eval_tools.py      # Model eval tool registry — 6 tools, Pydantic schemas
├── run_demo.py                  # 311 standalone runner — no Airflow needed
├── run_model_eval.py            # Model eval standalone runner — no Airflow needed
├── DEMO_STAGE_SCRIPT.sh         # Stage commands — open this before going on stage
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/itsChanelML/airclaw
cd airclaw
pip3 install -r requirements.txt
```

### 2. Get your NIM API key

1. Go to [build.nvidia.com](https://build.nvidia.com)
2. Sign in with your NVIDIA account
3. Search for `llama-3.3-nemotron-super-49b-v1`
4. Click **Get API Key** and copy it

```bash
cp .env.example .env
# Open .env and replace your_nim_api_key_here with your actual key
export NIM_API_KEY=your_key_here
```

### 3. Run the 311 triage demo

```bash
# Happy path — agent triages overnight data, drafts supervisor briefings
python3 run_demo.py

# Failure beat — schema drift triggers ESCALATE
python3 run_demo.py --break

# Audience input mode
python3 run_demo.py --goal "Which agency has the most overdue requests in Brooklyn?"
```

### 4. Run the model eval demo

```bash
# Happy path — agent compares GPT-4o vs Nemotron-Super, writes migration report
python3 run_model_eval.py

# Failure beat — eval schema drift triggers ESCALATE
python3 run_model_eval.py --break
```

### 5. Run with Airflow (optional — for showing the Airflow UI on stage)

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
mkdir -p $AIRFLOW_HOME/plugins $AIRFLOW_HOME/dags $AIRFLOW_HOME/tools

cp plugins/nemoclaw_operator.py  $AIRFLOW_HOME/plugins/
cp tools/airclaw_tools.py        $AIRFLOW_HOME/tools/
cp tools/model_eval_tools.py     $AIRFLOW_HOME/tools/
cp dags/airclaw_demo.py          $AIRFLOW_HOME/dags/
cp dags/model_eval_demo.py       $AIRFLOW_HOME/dags/

# Two terminals:
airflow webserver --port 8080
airflow scheduler

# Trigger either DAG:
airflow dags trigger airclaw_demo
airflow dags trigger model_eval_demo
```

---

## Pipeline 1 — 311 Triage Tools

| Tool | What it does | Error behavior |
|------|-------------|----------------|
| `validate_schema` | Verifies CSV has all required fields. Diagnoses renames if schema has drifted. | ESCALATE with exact diagnosis |
| `check_sla_breaches` | Finds every open request past its SLA window. Returns specific case IDs, districts, supervisors, hours overdue. | RETRY on parse error |
| `detect_complaint_spike` | Compares overnight volume to 7-day rolling baseline. Flags complaint types that jumped significantly. | RETRY on parse error |
| `prioritize_queue` | Ranks breaches by severity and clusters by geography. Identifies dispatch consolidation opportunities. | SUCCESS always |
| `draft_supervisor_briefing` | Writes a ready-to-send morning briefing per agency — ranked priorities, dispatch clusters, recommended actions. | ESCALATE if no agency/supervisor |
| `generate_summary` | One-paragraph duty manager overview. Final XCom output. | SUCCESS always |

---

## Pipeline 2 — Model Eval Tools

| Tool | What it does | Error behavior |
|------|-------------|----------------|
| `validate_schema` | Verifies eval CSV has all required fields. Diagnoses renames if schema has drifted. | ESCALATE with exact diagnosis |
| `score_comparison` | Compares Model A vs Model B quality and format scores by task category. | RETRY on parse error |
| `detect_regression` | Finds categories where Model B drops quality or spikes refusal rate. | RETRY on parse error |
| `cost_analysis` | Computes cost per prompt and latency delta. Projects monthly and annual savings. | RETRY on parse error |
| `draft_migration_report` | Writes a go/no-go recommendation with evidence. VP reads it in 5 min and makes a decision. | ESCALATE if no model names |
| `generate_summary` | One-paragraph summary of the eval run. Final XCom output. | SUCCESS always |

---

## Demo beats (stage guide)

### 311 Pipeline

**Beat 1 — Happy path**
```bash
python3 run_demo.py
```
Agent validates schema → finds 29 SLA breaches → detects spikes → ranks queue → drafts briefings with specific case IDs. Stop talking when the briefing streams. Let the room read it.

**Beat 2 — Failure beat**
```bash
python3 run_demo.py --break
```
`complaint_type` renamed to `complaint_category`. Agent diagnoses it precisely. ESCALATE. Clean failure, full audit trail.

### Model Eval Pipeline

**Beat 1 — Happy path**
```bash
python3 run_model_eval.py
```
Agent scores 300 prompts across 4 categories → detects regression on customer support → projects $9,272 annual savings → writes migration report: "PROCEED PARTIALLY — migrate code generation and RAG, hold customer support." Stop talking when the report streams.

**Beat 2 — Failure beat**
```bash
python3 run_model_eval.py --break
```
`model_b_quality_score` renamed to `model_b_score`. Migration report cannot be generated. ESCALATE with exact diagnosis.

---

## What makes it production-ready

**Typed tool contracts** — every tool takes a Pydantic schema in, returns a typed result out. SUCCESS, RETRY, or ESCALATE. No ambiguous strings, no freestyle.

**Idempotent execution** — the agent retries. Every tool is safe to call twice. Design for failure from day one.

**Structured escalation** — ESCALATE surfaces a typed diagnosis to Airflow, not a stack trace. The audit trail is clean. The human who picks it up gets exactly the context they need.

---

## The pattern generalizes

Any workflow where a human reviews pipeline output and decides what to do next is a candidate:

- Self-healing ETL — agent detects schema drift, adapts, keeps the pipeline running
- Intelligent incident triage — agent checks runbooks, takes corrective action, pages humans only when stuck
- Dynamic DAG branching — agent reads upstream output, chooses the right downstream path in real time
- On-call automation — agent handles the 2am page before it reaches a person
- Model governance — automated regression detection on every model update before it ships

You don't need to rebuild your stack. You need one operator and one goal.

---

## Built on

- **Apache Airflow** — orchestration, scheduling, observability
- **OpenClaw** — open-source agent runtime (tool contracts, reasoning loop, escalation)
- **NVIDIA NIM** — production inference for `nvidia/llama-3.3-nemotron-super-49b-v1`
- **Pydantic** — typed tool schemas

---

## About

Built by **Chanel Power** — Senior ML Engineer, Startup Advisor and Founder of [Mentor Me Collective](https://mentormecollective.org)

- GitHub: [@itsChanelML](https://github.com/itsChanelML)
- LinkedIn: [Chanel Power](https://linkedin.com/in/powerc1)
- Community: [mentormecollective.org](https://mentormecollective.org)

---

*Go build something that doesn't need you.*