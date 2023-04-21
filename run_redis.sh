#!/usr/bin/env bash

set -ex

PYTHON="${PYTHON:-python3}"
NS="${NS:-default}"
M="${M:-links}"

REDIS_PORT=$(${PYTHON} -m "system.namespace" port --namespace "${NS}" --module "${M}")
REDIS_PATH=$(${PYTHON} -m "system.namespace" path --namespace "${NS}" --module "${M}")

if [ "${NS}" = "_test" ]; then
    CFG=
else
    CFG=$(realpath "redis.main.conf")
fi

cd "${REDIS_PATH}" && redis-server "${CFG}" --port "${REDIS_PORT}"
