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
	@echo "lint-ts	run typescript linting"
	@echo "lint-ts-fix	run typescript linting with fix"
	@echo "pre-commit 	sort python package imports using isort"
	@echo "pytest	run all test with pytest (requires a running test redis server)"
	@echo "requirements-check	check whether the env differs from the requirements file"
	@echo "requirements-complete	check whether the requirements file is complete"
	@echo "run-test-redis	start redis server for pytest"
	@echo "run-redis	start redis server"
	@echo "run-api	start api server"
	@echo "run-web	start web server"

export LC_ALL=C
export LANG=C

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
	PYTHON=$(PYTHON) && ! ./forgottenformat.sh

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
	PYTHON=$(PYTHON) && ./install.sh

requirements-check:
	PYTHON=$(PYTHON) && ./requirements_check.sh $(FILE)

requirements-complete:
	PYTHON=$(PYTHON) && ./requirements_complete.sh $(FILE)

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
	MAKE=$(MAKE) && PYTHON=$(PYTHON) && RESULT_FNAME=$(RESULT_FNAME) && ./run_pytest.sh $(FILE)

run-test-redis:
	cd test && redis-server

run-redis:
	cd userdata && redis-server

run-api:
	python3 -m app

run-web:
	cd ui && yarn start

coverage-report:
	cd coverage/reports/html_report && open index.html
