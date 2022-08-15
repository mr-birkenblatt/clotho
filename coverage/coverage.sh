#!/usr/bin/env bash

set -e

coverage erase

find * \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
 -and -not -path './stubs/*' -exec python3 -m compileall -q -j 0 {} +

run_coverage() {
    pytest $1 --cov --cov-append
}
export -f run_coverage

for CUR in $(find 'test' \( -name '*.py' -and -name 'test_*' -and -not -name 'test_embedding*' \) \
            -and -not -path 'test/data/*' \
            -and -not -path 'test/__pycache__/*' |
            sort -sf); do
    run_coverage ${CUR}
done

coverage xml -o coverage/reports/xml_report.xml
coverage html -d coverage/reports/html_report
coverage erase
