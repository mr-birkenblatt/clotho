import argparse
import os

from app.server import setup_server, start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {os.path.basename(os.path.dirname(__file__))}",
        description="Run the API server")
    parser.add_argument(
        "--namespace",
        default=None,
        help="the namespace of the API server")
    parser.add_argument(
        "--address",
        default=None,
        help="the address of the API server")
    parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="the port of the API server")
    parser.add_argument(
        "--dedicated",
        default=False,
        action="store_true",
        help="the port of the API server")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    server, prefix = setup_server(
        deploy=args.dedicated,
        ns_name=args.namespace,
        addr=args.address,
        port=args.port)
    start(server, prefix)


if __name__ == "__main__":
    run()
