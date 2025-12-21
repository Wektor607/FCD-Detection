# #!/bin/bash

# #source FreeSurfer 
# source $FREESURFER_HOME/FreeSurferEnv.sh
# # Run your application
# # exec gosu myuser python "scripts/new_patient_pipeline/$@"
# # python scripts/new_patient_pipeline/$@
# $@

#!/usr/bin/env bash
set -e

if [ -z "$FREESURFER_HOME" ]; then
  echo "ERROR: FREESURFER_HOME is not set"
  exit 1
fi

if [ ! -f "$FREESURFER_HOME/FreeSurferEnv.sh" ]; then
  echo "ERROR: $FREESURFER_HOME/FreeSurferEnv.sh not found"
  exit 1
fi

source "$FREESURFER_HOME/FreeSurferEnv.sh"

exec "$@"
