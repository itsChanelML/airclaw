#!/usr/bin/env bash
#
# AirClaw — one-command Airflow 3 startup
# --------------------------------------------------------------------
# Points Airflow at this repo's dags/ and plugins/ directories, so there
# is nothing to copy and nothing to keep in sync. Then starts the API
# server and scheduler together.
#
#   ./run_airflow.sh          # start Airflow, UI on http://localhost:8080
#   ./run_airflow.sh --check  # parse DAGs and exit (no server)
#
# Requires NIM_API_KEY in .env for the agent tasks to run.
# --------------------------------------------------------------------

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AIRCLAW_HOME="$REPO"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$REPO/airflow_home}"
export AIRFLOW__CORE__DAGS_FOLDER="$REPO/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="$REPO/plugins"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# Demo DAGs are triggered by hand on stage; don't let the 6am schedule
# backfill a year of runs the moment the scheduler starts.
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True

mkdir -p "$AIRFLOW_HOME"

echo "AIRFLOW_HOME : $AIRFLOW_HOME"
echo "DAGs         : $AIRFLOW__CORE__DAGS_FOLDER"
echo "Plugins      : $AIRFLOW__CORE__PLUGINS_FOLDER"
echo

# `db migrate` is idempotent — safe on every start, and it replaces the
# `airflow db init` from Airflow 2 which no longer exists.
echo "→ Migrating metadata database…"
airflow db migrate >/dev/null 2>&1
echo "  done."

echo "→ Parsing DAGs…"
airflow dags reserialize >/dev/null 2>&1
if [ -n "$(airflow dags list-import-errors 2>/dev/null | grep -v 'No data found' | grep -v '^$' || true)" ]; then
  echo "  DAG import errors:"
  airflow dags list-import-errors
  exit 1
fi
airflow dags list 2>/dev/null | grep -E "airclaw_demo|model_eval_demo" || true
echo "  both DAGs parsed cleanly."

if [ "${1:-}" = "--check" ]; then
  echo
  echo "Check complete. Run without --check to start the server."
  exit 0
fi

# Unpause so the DAGs can be triggered from the UI without a click first.
airflow dags unpause airclaw_demo    >/dev/null 2>&1 || true
airflow dags unpause model_eval_demo >/dev/null 2>&1 || true

echo
echo "→ Starting Airflow. UI: http://localhost:8080"
echo "  Trigger from the UI, or in another terminal:"
echo "    airflow dags trigger airclaw_demo"
echo "    airflow dags trigger model_eval_demo"
echo "  Ctrl-C to stop."
echo

# `airflow standalone` runs api-server + scheduler + dag-processor together
# and prints admin credentials on first start. In Airflow 3 the Airflow 2
# `airflow webserver` command no longer exists.
exec airflow standalone
