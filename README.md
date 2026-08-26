# AirClaw 🦾
### Autonomous Self-Healing Agents with Apache Airflow + NemoClaw
**DevFest DC · August 2026**

> "What if your pipeline could think for itself?"

---

## What this is

AirClaw pairs **Apache Airflow** (orchestration) with **NemoClaw** — an agent loop running `nvidia/nemotron-3-super-120b-a12b` via NVIDIA NIM, built on [OpenClaw](https://github.com/openclawai/openclaw)-style typed tool contracts — to create self-healing pipelines that reason, adapt, and fail gracefully without human intervention.

The agent runtime is deliberately small: a tool-calling loop against the NIM API, a typed registry, and a `SUCCESS / RETRY / ESCALATE` contract. No framework to learn before you can read it.

Two pipelines. Same framework. Two different problems.

**Pipeline 1 — NYC 311 Triage Agent**
Every morning a city agency supervisor opens their laptop to 300 overnight 311 service requests. Some are overdue past their SLA window. Some are complaint spikes. Someone needs to find all of that, figure out who to call, and write the briefing. That someone is a person — and it takes them 30 to 60 minutes every single morning. AirClaw does it instead. The supervisor's first action of the day is approving work the pipeline already did.

**Pipeline 2 — Model Migration Eval Agent**
A team wants to migrate from GPT-4o to Nemotron 3 Super in production. Someone needs to run both models on production prompts, compare quality by task category, detect regressions, project cost savings, and write a go/no-go recommendation. That analysis takes a Sr Engineer 2-3 days manually. AirClaw does it in one pipeline run.

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
- **NemoClaw** — a NIM tool-calling loop over `nvidia/nemotron-3-super-120b-a12b` (set `NIM_MODEL` to use another). Owns the how: it picks the tools and the order.
- **Tool Registry** — typed tools with `SUCCESS / RETRY / ESCALATE` contracts. No silent failures.

---

## Project structure

```
airclaw/
├── dags/
│   ├── airclaw_demo.py          # 311 triage DAG — 3 tasks, 6am schedule
│   └── model_eval_demo.py       # Model eval DAG — triggered on eval completion
├── data/
│   ├── nyc_311_clean.csv        # Happy path — 300 requests, 29 SLA breaches (dates rebased at run time)
│   ├── nyc_311_broken.csv       # Broken schema — complaint_type → complaint_category
│   ├── nyc_311_upstream.csv     # Active file — swapped by run_demo.py automatically
│   ├── model_eval_clean.csv     # Happy path — 300 prompts, GPT-4o vs Nemotron 3 Super
│   ├── model_eval_broken.csv    # Broken schema — model_b_quality_score → model_b_score
│   └── model_eval_upstream.csv  # Active file — swapped by run_model_eval.py automatically
├── plugins/
│   └── nemoclaw_operator.py     # Custom Airflow operator + plain-English log formatter
├── tools/
│   ├── airclaw_tools.py         # 311 tool registry — 6 tools, Pydantic schemas
│   ├── schema_diff.py           # Shared schema-drift / rename diagnosis
│   └── model_eval_tools.py      # Model eval tool registry — 6 tools, Pydantic schemas
├── airclaw_env.py               # .env loading, API key validation, path resolution
├── rebase_data.py               # Shifts sample timestamps onto today (see below)
├── preflight.py                 # Pre-show check — run this before you present
├── run_demo.py                  # 311 standalone runner — no Airflow needed
├── run_model_eval.py            # Model eval standalone runner — no Airflow needed
├── run_airflow.sh               # One-command Airflow 3 startup (--check to parse only)
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
3. Search for `nemotron-3-super-120b-a12b`
4. Click **Get API Key** and copy it

```bash
cp .env.example .env
# Open .env and replace your_nim_api_key_here with your actual key
```

Both runners and the Airflow operator load `.env` automatically. If the key is
missing, still the placeholder, or malformed, you get a one-line diagnosis
instead of a confusing 401 from NIM.

**Model EOL.** NVIDIA retires hosted models on published end-of-life dates, and
a retired model returns `HTTP 410 Gone` on every call — this project's original
model, `llama-3.3-nemotron-super-49b-v1`, went EOL on 2026-08-26. The model id
lives in one place (`airclaw_env.get_model()`) and is overridable with
`NIM_MODEL` in `.env`. To see what is currently served:

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NIM_API_KEY" | grep -o '"id":"[^"]*"'
```

### 3. Check you're ready

```bash
python3 preflight.py
```

Verifies dependencies, the API key, that the model is actually **served** (a
live tool call — this is what catches a model retirement), the data files, the
rebasing invariants, tool-registry parity, and that both DAGs parse. Exits
non-zero if anything would fail on stage.

### 4. Run the 311 triage demo

```bash
# Happy path — agent triages overnight data, drafts supervisor briefings
python3 run_demo.py

# Failure beat — schema drift triggers ESCALATE
python3 run_demo.py --break

# Audience input mode — the agent plans its own approach and answers
python3 run_demo.py --goal "Which agency has the most overdue requests in Brooklyn?"
```

In audience-input mode the agent chooses its own tools: it reaches for
`query_requests` to gather the facts and replies with a direct answer instead of
running the full triage. Tool selection and ordering are the agent's decisions —
the goal states what to accomplish, not which functions to call.

### 5. Run the model eval demo

```bash
# Happy path — agent compares GPT-4o vs Nemotron-Super, writes migration report
python3 run_model_eval.py

# Failure beat — eval schema drift triggers ESCALATE
python3 run_model_eval.py --break
```

### 6. Run with Airflow (optional — for showing the Airflow UI on stage)

```bash
./run_airflow.sh --check   # parse both DAGs and exit — pre-flight check
./run_airflow.sh           # start Airflow, UI on http://localhost:8080
```

The script points Airflow at this repo's `dags/` and `plugins/` directories, so
there is nothing to copy and nothing to keep in sync. It runs `airflow db
migrate`, verifies both DAGs parse, and starts `airflow standalone`
(api-server + scheduler + dag-processor together).

Trigger either pipeline:

```bash
airflow dags trigger airclaw_demo
airflow dags trigger model_eval_demo
```

Briefings and migration reports stream into the task logs in real time — the
Airflow UI works as a demo surface, not just a status board.

> Built against **Airflow 3**. The Airflow 2 commands (`airflow db init`,
> `airflow webserver`) no longer exist; `run_airflow.sh` uses the current ones.

---

## Always-current data

The 311 sample carries fixed timestamps, but every SLA and spike window is
relative to `datetime.now()`. Left alone, the demo decays: spike detection goes
quiet, the ingest task reports zero overnight requests, and case IDs date
themselves.

`rebase_data.py` shifts the sample onto the present every time a runner or the
DAG's ingest task loads it — the newest request becomes "now", `hours_open`
stays true to `created_date`, and case ID years follow. It also spreads the
movable records across the baseline week so overnight volume reads as a
plausible surge rather than the entire file.

What it must never change is asserted after every rebase, and raises rather than
writing data that contradicts the demo:

- the 29 SLA breaches, their agencies, supervisors and hours overdue
- closed requests' resolution durations
- `hours_open`'s agreement with `created_date`
- no request created or closed in the future

Breach rows are never moved. Everything else carries bounds: an open request
can't age past its SLA without becoming a breach, and a closed request can't be
younger than the time it took to resolve.

---

## Pipeline 1 — 311 Triage Tools

| Tool | What it does | Error behavior |
|------|-------------|----------------|
| `validate_schema` | Verifies CSV has all required fields. Diagnoses renames if schema has drifted. | ESCALATE with exact diagnosis |
| `check_sla_breaches` | Finds every open request past its SLA window. Returns specific case IDs, districts, supervisors, hours overdue. | RETRY on parse error |
| `detect_complaint_spike` | Compares overnight volume to a 7-day rolling baseline. Flags complaint types that jumped significantly, ranked by how many extra requests arrived rather than by percentage, and ignores types below a volume floor so a jump from 0.3/day to 4 doesn't crowd out the real cluster. | RETRY on parse error |
| `query_requests` | Answers counting questions about the feed — group by complaint type, borough, district, agency, status, or supervisor, with filters like `only_breaches`. Reports ties explicitly. This is what makes audience-input mode work. | RETRY on bad `group_by` |
| `draft_supervisor_briefing` | Writes a ready-to-send morning briefing for one agency — ranked priorities, dispatch clusters, recommended actions. Reads case data from the file itself, so the agent passes only agency, supervisor, and file path. | ESCALATE if no agency/supervisor |
| `generate_summary` | One-paragraph duty manager overview. Final XCom output. | SUCCESS always |

---

## Pipeline 2 — Model Eval Tools

| Tool | What it does | Error behavior |
|------|-------------|----------------|
| `validate_schema` | Verifies eval CSV has all required fields. Diagnoses renames if schema has drifted. | ESCALATE with exact diagnosis |
| `score_comparison` | Compares Model A vs Model B quality and format scores by task category. | RETRY on parse error |
| `detect_regression` | Finds categories where Model B drops quality or spikes refusal rate. | RETRY on parse error |
| `cost_analysis` | Computes cost per prompt and latency delta. Projects monthly and annual savings. | RETRY on parse error |
| `draft_migration_report` | Writes a go/no-go recommendation with evidence. Recomputes the analysis from the eval file, so the agent passes only the two model names and the file path — no echoing payloads back through message history. | ESCALATE if no model names or no eval data |
| `generate_summary` | One-paragraph summary of the eval run. Final XCom output. | SUCCESS always |

---

## Demo beats (stage guide)

### 311 Pipeline

**Beat 1 — Happy path**
```bash
python3 run_demo.py
```
Agent validates schema → finds 29 SLA breaches → detects 3 overnight complaint spikes (Illegal Parking up 271%, Noise-Residential up 232%, Blocked Driveway up 133%) → picks the agencies carrying the most breaches → drafts briefings with specific case IDs, real supervisor names, and the spike alerts that belong to that agency. Stop talking when the briefing streams. Let the room read it.

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
Agent scores 300 prompts across 4 categories → detects regressions on RAG QA and customer support → projects $9,272 annual savings (66% cheaper, 50% faster) → writes migration report: "PROCEED PARTIALLY — migrate code generation. Hold RAG QA and customer support." Stop talking when the report streams.

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
- **OpenClaw** — the tool-contract pattern this agent loop follows (typed tools, reasoning loop, structured escalation)
- **NVIDIA NIM** — production inference for `nvidia/nemotron-3-super-120b-a12b`
- **Pydantic** — typed tool schemas

---

## About

Built by **Chanel Power** — Senior ML Engineer, Startup Advisor and Founder of [Mentor Me Collective](https://mentormecollective.org)

- GitHub: [@itsChanelML](https://github.com/itsChanelML)
- LinkedIn: [Chanel Power](https://linkedin.com/in/powerc1)
- Community: [mentormecollective.org](https://mentormecollective.org)

---

*Go build something that doesn't need you.*
