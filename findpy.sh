find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './userdata' -o \
        -path './stubs_pre' -o \
        -path './ui' \
        \) -prune -o \( \
        -name '*.py' -o \
        -name '*.pyi' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './userdata' \
    | grep -vF './stubs_pre' \
    | grep -vF './ui'
