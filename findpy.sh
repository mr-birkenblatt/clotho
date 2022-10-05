find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './node_modules' -o \
        -path './userdata' \
        \) -prune -o \( \
        -name '*.py' -o \
        -name '*.pyi' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './node_modules' \
    | grep -vF './userdata'
