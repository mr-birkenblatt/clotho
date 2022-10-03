#!/usr/bin/env bash

set -ex

PYTHON="${PYTHON:-python3}"
FILES=($@)
export USER_FILEPATH=./userdata

find . -type d \( \
    -path './venv' -o \
    -path './.*' -o \
    -path './stubs' -o \
    -path './node_modules' -o \
    -path './userdata' \
    \) -prune -o \
    -name '*.py' \
    -exec ${PYTHON} -m compileall -q -j 0 {} +

redis-cli \
    "EVAL" \
    "for _,k in ipairs(redis.call('keys', KEYS[1])) do redis.call('del', k) end" \
    1 \
    'api:test:*'

run_test() {
    ${PYTHON} -m pytest -xvv --full-trace --junitxml="test-results/parts/result${2}.xml" $1
}
export -f run_test

if ! [ -z ${FILES} ]; then
    IDX=0
    for CUR_TEST in ${FILES[@]}; do
        run_test $CUR_TEST $IDX
        IDX=$((IDX+1))
    done
else
    IDX=0
    for CUR in $(find 'test' \( -name '*.py' -and -name 'test_*' \) \
            -and -not -path 'test/data/*' \
            -and -not -path 'test/__pycache__/*' |
            sort -sf); do
        run_test ${CUR} $IDX
        IDX=$((IDX+1))
    done
fi
${PYTHON} -c "from test.util import merge_results; merge_results('./test-results')"
rm -r test-results/parts
