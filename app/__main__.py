from app.server import setup_server, start


def run() -> None:
    server, prefix = setup_server(deploy=False, addr=None, port=None)
    start(server, prefix)


if __name__ == "__main__":
    run()
