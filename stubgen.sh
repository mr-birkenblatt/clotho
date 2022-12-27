#!/usr/bin/env bash

set -ex

PYTHON="${PYTHON:-python3}"
STUBGEN="${STUBGEN:-stubgen}"
OUTPUT="${OUTPUT:-stubs}"
PACKAGE="${1}"
FULL_OUTPUT="${OUTPUT}/${PACKAGE}"
TMP_FILE="~tmp"
STUBGEN_HEAD="stubgen.head"

if [ -z "${PACKAGE}" ]; then
    echo "usage: ${0} <package>"
    exit 1
fi

${STUBGEN} -p "${PACKAGE}" -o "${OUTPUT}"

if [ -f "${TMP_FILE}" ]; then
    echo "${TMP_FILE} already exists! cannot use as tmp file"
    exit 1
fi

find "${FULL_OUTPUT}" -name '*.pyi' \
    -exec echo \
        "mv {} ${TMP_FILE}" \
        "&& cp ${STUBGEN_HEAD} {}" \
        "&& ${PYTHON} stubgen.py ${TMP_FILE} {}" \
        "&& rm ${TMP_FILE}" \; \
    | sh
