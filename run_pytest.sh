#!/usr/bin/env bash

set -ex

PYTHON="${PYTHON:-python3}"
RESULT_FNAME="${RESULT_FNAME:-results.xml}"
FILES=($@)
export USER_FILEPATH=./userdata

coverage erase

find . -type d \( \
    -path './venv' -o \
    -path './.*' -o \
    -path './stubs' -o \
    -path './node_modules' -o \
    -path './userdata' \
    \) -prune -o \
    -name '*.py' \
    -exec ${PYTHON} -m compileall -q -j 0 {} +

if command -v redis-cli &> /dev/null; then
    redis-cli \
        "EVAL" \
        "for _,k in ipairs(redis.call('keys', KEYS[1])) do redis.call('del', k) end" \
        1 \
        'api:test:*'
fi

run_test() {
    ${PYTHON} -m pytest \
        -xvv --full-trace \
        --junitxml="test-results/parts/result${2}.xml" \
        --cov --cov-append \
        $1
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
    for CUR in $(find 'test' -type d \( \
            -path 'test/data' -o \
            -path 'test/__pycache__' \
            \) -prune -o \( \
            -name '*.py' -and \
            -name 'test_*' \
            \) | \
            grep -E '.*\.py' | \
            sort -sf); do
        run_test ${CUR} $IDX
        IDX=$((IDX+1))
    done
fi
${PYTHON} -m test merge_results --dir test-results --out-fname ${RESULT_FNAME}
rm -r test-results/parts

coverage xml -o coverage/reports/xml_report.xml
coverage html -d coverage/reports/html_report
