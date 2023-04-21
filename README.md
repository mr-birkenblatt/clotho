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

Use `python -m system.namespace xfer --namespace <ns_src> --module <module> --dest <ns_dest>`
to transfer data between namespaces (you can use this to change the module type, too).

Use `python -m system.namespace init --namespace <namespace> --module <module>` to initialize a module.
Not all modules need initialization.

Registering a model:
`python -m model.register --namespace <namespace> --connection <conn> --file <modelfile> --version <modelversion> [--is-harness]`
The namespace is only used to obtain the database connection `<conn>`.
This is necessary to use db stored embeddings for lookups.
