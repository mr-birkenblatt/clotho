#!/usr/bin/env bash

set -e

REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_CMD="${REDIS_CMD:-redis-cli}"
STEP="${STEP:-1000}"

CURSOR=0
while true; do
    IS_FIRST=1
    for LINE in $("${REDIS_CMD}" -p "${REDIS_PORT}" SCAN "${CURSOR}" MATCH '*' COUNT "${STEP}"); do
        if [ ! -z "${IS_FIRST}" ]; then
            CURSOR="${LINE}"
            IS_FIRST=
        else
            echo "[key]"
            echo "${LINE}"
            TYPE=$("${REDIS_CMD}" -p "${REDIS_PORT}" TYPE "${LINE}")
            SIZE=
            case ${TYPE} in
                string)
                    OUT=$("${REDIS_CMD}" -p "${REDIS_PORT}" GET "${LINE}")
                    ;;
                list)
                    SIZE=:$("${REDIS_CMD}" -p "${REDIS_PORT}" LLEN "${LINE}")
                    OUT=$("${REDIS_CMD}" -p "${REDIS_PORT}" LRANGE "${LINE}" 0 -1)
                    ;;
                set)
                    SIZE=:$("${REDIS_CMD}" -p "${REDIS_PORT}" SCARD "${LINE}")
                    OUT=$("${REDIS_CMD}" -p "${REDIS_PORT}" SMEMBERS "${LINE}")
                    ;;
                *)
                    echo "ERROR: ${TYPE}"
                    exit 1
                    ;;
            esac
            echo "[${TYPE}${SIZE}]"
            echo "${OUT}"
        fi
    done
    if [ $CURSOR = 0 ]; then
        break
    fi
done
