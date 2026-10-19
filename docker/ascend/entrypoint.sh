#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh" ]]; then
    source "$ASCEND_HOME_PATH/share/info/ascendnpu-ir/bin/set_env.sh"
fi
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
python3 -m pip install --user --no-deps --no-build-isolation \
    -e /workspace/assignment5-alignment
exec "$@"
