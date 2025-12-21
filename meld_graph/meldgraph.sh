# #!/bin/bash

# export MELD_LICENSE='meld_license.txt'

# eval "$(conda shell.bash hook)"
# conda activate meld_graph

# if [ $1 = 'pytest' ]; then
#   pytest ${@:2}
# else
#   python scripts/new_patient_pipeline/$1 ${@:2}
# fi

#!/bin/bash
set -e

export MELD_LICENSE='meld_license.txt'

# ⚠️ conda ТОЛЬКО если она реально есть
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
#   conda activate meld_graph
fi

SCRIPT_NAME="${1%.py}"
shift

python -m scripts.new_patient_pipeline.${SCRIPT_NAME} "$@"
