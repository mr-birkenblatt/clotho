help:
	@echo "The following make targets are available:"
	@echo "install	install all dependencies"
	@echo "lint-emptyinit	main inits must be empty"
	@echo "lint-flake8	run flake8 checker to deteck missing trailing comma"
	@echo "lint-forgottenformat	ensures format strings are used"
	@echo "lint-indent	run indent format check"
	@echo "lint-pycodestyle	run linter check using pycodestyle standard"
	@echo "lint-pycodestyle-debug	run linter in debug mode"
	@echo "lint-pyi	Ensure no regular python files exist in stubs"
	@echo "lint-pylint	run linter check using pylint standard"
	@echo "lint-requirements	run requirements check"
	@echo "lint-stringformat	run string format check"
	@echo "lint-type-check	run type check"
	@echo "pre-commit 	sort python package imports using isort"
	@echo "pytest	run all test with pytest"
	@echo "requirements-check	check whether the env differs from the requirements file"
	@echo "requirements-complete	check whether the requirements file is complete"
	@echo "run-redis	start redis server"

export LC_ALL=C
export LANG=C

lint-comment:
	! find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| xargs grep --color=always -nE \
	  '#.*(todo|xxx|fixme|n[oO][tT][eE]:|Note:|nopep8\s*$$)|.\"^s%'

lint-emptyinit:
	[ ! -s app/__init__.py ]

lint-pyi:
	./pyi.sh

lint-stringformat:
	! find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| xargs grep --color=always -nE "%[^'\"]*\"\\s*%\\s*"

lint-indent:
	! find . \( -name '*.py' -or -name '*.pyi' \) -and -not -path './venv/*' \
	| xargs grep --color=always -nE "^(\s{4})*\s{1,3}\S.*$$"

lint-forgottenformat:
	! ./forgottenformat.sh

lint-requirements:
	locale
	cat requirements.txt
	sort -ufc requirements.txt

lint-pycodestyle:
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort | xargs pycodestyle --exclude=venv --show-source

lint-pycodestyle-debug:
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort | xargs pycodestyle --exclude=venv,.git,.mypy_cache -v --show-source

lint-pylint:
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort
	find . \( -name '*.py' -o -name '*.pyi' \) -and -not -path './venv/*' \
	| sort | xargs pylint -j 6

lint-type-check:
	mypy . --config-file mypy.ini

lint-flake8:
	flake8 --verbose --select C812,C815,I001,I002,I003,I004,I005 --exclude \
	venv --show-source ./

lint-all: \
	lint-comment \
	lint-emptyinit \
	lint-pyi \
	lint-stringformat \
	lint-indent \
	lint-forgottenformat \
	lint-requirements \
	requirements-complete \
	lint-pycodestyle \
	lint-pylint \
	lint-type-check \
	lint-flake8

install:
	./install.sh $(PYTHON)

requirements-check:
	./requirements_check.sh $(PYTHON) $(FILE)

requirements-complete:
	./requirements_complete.sh $(PYTHON) $(FILE)

name:
	git describe --abbrev=10 --tags HEAD

git-check:
	./git_check.sh

pre-commit:
	pre-commit install
	isort .

coverage-report:
	./coverage/coverage.sh

run-redis:
	cd userdata && redis-server
