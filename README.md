# clotho

To get started with a given namespace after `make install` and `make install-ts`
run in separate terminals:

```
make run-redis-api
make run-redis NS=<namespace>  # if the namespace redis is different (recommended)
make run-api NS=<namespace>
make run-web
```

To run tests and lints start the test redis `make run-redis-test` and run:

```
make lint-all  # linting does not require redis
make pytest
```

Typescript tests don't need redis:

```
make lint-ts
make ts-unused
make test-ts
```

Use `make pre-commit` to set up git to enable commit checks.

`python -m example reddit <namespace>` to crawl reddit and
`python -m example load <namespace>` to load it into the given namespace
(requires a running redis for the namespace)

`python -m model` to check whether train / test generation is working
