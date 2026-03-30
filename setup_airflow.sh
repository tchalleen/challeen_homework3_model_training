#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------
# Airflow Lab Setup Script (project-isolated)
# -------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
AIRFLOW_HOME_DIR="$PROJECT_ROOT/airflow_home"

export AIRFLOW_HOME="$AIRFLOW_HOME_DIR"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="$PROJECT_ROOT/plugins"
export AIRFLOW__LOGGING__BASE_LOG_FOLDER="$AIRFLOW_HOME/logs"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$AIRFLOW_HOME/airflow.db"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export S3_BUCKET_ARN="arn:aws:s3:::st-mlops-tony-challeen-bucket"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "AIRFLOW_HOME=$AIRFLOW_HOME"

# Create local folders
mkdir -p "$AIRFLOW_HOME"
mkdir -p "$PROJECT_ROOT/dags"
mkdir -p "$PROJECT_ROOT/plugins"
mkdir -p "$AIRFLOW_HOME/logs"

# Use the external venv with Python 3.11
EXTERNAL_VENV="$HOME/venvs/airflow-class"
if [[ -d "$EXTERNAL_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$EXTERNAL_VENV/bin/activate"
  echo "Activated virtual environment: $EXTERNAL_VENV"
else
  echo "Error: No virtual environment found at $EXTERNAL_VENV"
  echo "Please run: python3.11 -m venv ~/venvs/airflow-class && source ~/venvs/airflow-class/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | grep -o '3\.11')
if [[ -z "$PYTHON_VERSION" ]]; then
  echo "Error: Expected Python 3.11, got $(python --version)"
  echo "Please recreate the venv with Python 3.11"
  exit 1
fi

# Initialize Airflow metadata DB locally
python -m airflow db init

# Create admin user if it doesn't exist
python -m airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin || echo "Admin user already exists"

echo
echo "Airflow lab environment is set up locally."
echo "Nothing was written to ~/.bashrc, ~/.zshrc, or ~/.profile."
echo
echo "To use this environment in the current shell:"
echo "  source ./setup_airflow.sh"
echo
echo "Then run:"
echo "  python -m airflow webserver --port 8080 --host 0.0.0.0"
echo "  python -m airflow scheduler"