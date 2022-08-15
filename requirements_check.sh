#!/usr/bin/env bash

PYTHON=${1:-python3}
FILE=${2:-requirements.txt}

${PYTHON} -m pip freeze | sort -sf | grep -i -E "^($(cat ${FILE} | sed -E 's/[=~]=.+//' | perl -p -e 'chomp if eof' | tr '\n' '|'))=" | diff -U 0 ${FILE} -

echo "NOTE: '+' is your local version and '-' is the version in ${FILE}" 1>&2
