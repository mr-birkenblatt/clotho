help:
	@echo "The following make targets are available:"
	@echo "install	install all python dependencies"
	@echo "install-ts	install all typescript dependencies"
	@echo "lint-comment	ensures fixme comments are grepable"
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
	@echo "lint-ts	run typescript linting"
	@echo "lint-ts-fix	run typescript linting with fix"
	@echo "lint-all	run all lints"
	@echo "pre-commit 	sort python package imports using isort"
	@echo "name	generate a unique permanent name for the current commit"
	@echo "git-check	ensures no git visible files have been altered"
	@echo "clean	clean test data"
	@echo "pytest	run all test with pytest (requires a running test redis server)"
	@echo "test-ts	run all typescript tests"
	@echo "ts-unused	check for unused exports in typescript"
	@echo "ts-build	build the ui code"
	@echo "requirements-check	check whether the env differs from the requirements file"
	@echo "requirements-complete	check whether the requirements file is complete"
	@echo "run-redis-test	start redis server for pytest"
	@echo "run-redis-api	start redis server for api (note, this is separate from redis required by modules)"
	@echo "run-redis	start redis server"
	@echo "run-api	start api server"
	@echo "run-web	start web server"
	@echo "coverage-report	show the coverage report for python"
	@echo "coverage-report-ts	show the coverage report for typescript"
	@echo "stubgen	create stubs for a package"

export LC_ALL=C
export LANG=C

PYTHON=python3
NS=default

lint-comment:
	! ./findpy.sh \
	| xargs grep --color=always -nE \
	  '#.*(todo|xxx|fixme|n[oO][tT][eE]:|Note:|nopep8\s*$$)|.\"^s%'

lint-emptyinit:
	[ ! -s app/__init__.py ]

lint-pyi:
	./pyi.sh

lint-stringformat:
	! ./findpy.sh \
	| xargs grep --color=always -nE "%[^'\"]*\"\\s*%\\s*"

lint-indent:
	! ./findpy.sh \
	| xargs grep --color=always -nE "^(\s{4})*\s{1,3}\S.*$$"

lint-forgottenformat:
	! PYTHON=$(PYTHON) ./forgottenformat.sh

lint-requirements:
	locale
	cat requirements.txt
	sort -ufc requirements.txt

lint-pycodestyle:
	./findpy.sh | sort
	./findpy.sh | sort | xargs pycodestyle --exclude=venv --show-source

lint-pycodestyle-debug:
	./findpy.sh | sort
	./findpy.sh \
	| sort | xargs pycodestyle --exclude=venv,.git,.mypy_cache -v --show-source

lint-pylint:
	./findpy.sh | sort
	./findpy.sh | sort | xargs pylint -j 6

lint-type-check:
	mypy . --config-file mypy.ini

lint-flake8:
	flake8 --verbose --select C812,C815,I001,I002,I003,I004,I005 --exclude \
	venv --show-source ./

lint-ts:
	cd ui && yarn lint

lint-ts-fix:
	cd ui && yarn lint --fix

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
	lint-flake8 \
	lint-ts

install:
	PYTHON=$(PYTHON) ./install.sh

install-ts:
	cd ui && yarn install

requirements-check:
	PYTHON=$(PYTHON) ./requirements_check.sh $(FILE)

requirements-complete:
	PYTHON=$(PYTHON) ./requirements_complete.sh $(FILE)

name:
	git describe --abbrev=10 --tags HEAD

git-check:
	./git_check.sh

pre-commit:
	pre-commit install
	isort .

clean:
	./clean.sh

pytest:
	MAKE=$(MAKE) PYTHON=$(PYTHON) RESULT_FNAME=$(RESULT_FNAME) ./run_pytest.sh $(FILE)

test-ts:
	cd ui && yarn testall

ts-unused:
	cd ui && yarn unused

ts-build:
	cd ui && yarn build

run-redis-test:
	PYTHON=$(PYTHON) NS=_test ./run_redis.sh

run-redis-api:
	PYTHON=$(PYTHON) NS=_api ./run_redis.sh

run-redis:
	PYTHON=$(PYTHON) NS=$(NS) ./run_redis.sh

run-api:
	API_SERVER_NAMESPACE=$(NS) $(PYTHON) -m app

run-web:
	cd ui && yarn start

coverage-report:
	cd coverage/reports/html_report && open index.html

coverage-report-ts:
	cd ui/coverage/lcov-report && open index.html

stubgen:
	PYTHON=$(PYTHON) ./stubgen.sh $(PKG)
