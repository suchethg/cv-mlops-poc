# Set up a terminal tab for working on this project.
# Usage (from the project folder):  source scripts/env.sh
#
# Does what `metaflow-dev shell` does (points Metaflow at the local stack),
# plus activates the virtual environment and loads the data lake credentials.
# Requires `metaflow-dev up` to be running in another tab.

conda deactivate 2>/dev/null
conda deactivate 2>/dev/null
source .venv/bin/activate

export METAFLOW_HOME="$PWD/.venv/share/metaflow/devtools/.devtools"
export METAFLOW_PROFILE=local
export AWS_CONFIG_FILE="$METAFLOW_HOME/aws_config"
export AWS_SHARED_CREDENTIALS_FILE=
unset AWS_PROFILE

set -a; source .env.local; set +a

if [ -f "$METAFLOW_HOME/config_local.json" ]; then
  echo "ready: Metaflow -> local stack, data lake credentials loaded"
else
  echo "warning: $METAFLOW_HOME/config_local.json not found. Is 'metaflow-dev up' running?"
fi