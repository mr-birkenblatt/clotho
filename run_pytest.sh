#!/usr/bin/env bash

set -e

PYTHON="${PYTHON:-python3}"
FILES=($@)
export USER_FILEPATH=./userdata

find . -name '*.py' -and -not -path './venv/*' \
-and -not -path './stubs/*' -exec ${PYTHON} -m compileall -q -j 0 {} +

run_test() {
    ${PYTHON} -m pytest -xvv --full-trace --junitxml="test-results/parts/result${2}.xml" $1
}
export -f run_test

redis-cli --scan --pattern 'api:test:*' | xargs redis-cli del

if ! [ -z ${FILES} ]; then
    IDX=0
    for CUR_TEST in ${FILES[@]}; do
        run_test $CUR_TEST $IDX
        IDX=$((IDX+1))
    done
    ${PYTHON} -c "from test.util import merge_results; merge_results('./test-results')"
    rm -r test-results/parts
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
