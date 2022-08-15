#!/usr/bin/env bash

PYTHON=${1:-python3}
FILE=${2:-requirements.txt}

cat "${FILE}" | sed -E 's/[=~]=.+//' | sort -sf | diff -U 1 "requirements.noversion.txt" -
