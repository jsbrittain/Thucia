import argparse
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .cli import CommandsList

try:
    __version__ = version("thucia")
except PackageNotFoundError:
    # package is not installed
    pass


def build_parser() -> argparse.ArgumentParser:
    # Parser
    parser = argparse.ArgumentParser(description="Thucia")
    sub = parser.add_subparsers(
        dest="_command",
        title="Available Commands",
        metavar="<command>",
        required=True,
    )
    for c in CommandsList:
        c.add_to_subparsers(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd_cls = getattr(args, "_command_class", None)
    if cmd_cls is None:
        parser.print_help()
        return 2
    cmd = cmd_cls()
    try:
        return cmd.execute(args)
    except Exception as exc:
        print("Unhandled error:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
