#!/usr/bin/env bash

set -ex

PYTHON="${PYTHON:-python3}"
NS="${NS:-default}"
USER_PATH="${USER_PATH:-userdata}"

PORT=$(${PYTHON} -m "system.namespace" port --namespace "${NS}")
if [ "${NS}" = "_test" ]; then
    CFG=
    USER_PATH="test"
else
    CFG="../redis.main.conf"
fi

cd "${USER_PATH}" && redis-server "${CFG}" --port "${PORT}"
