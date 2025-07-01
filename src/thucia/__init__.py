import argparse
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("thucia")
except PackageNotFoundError:
    # package is not installed
    pass


def launch_dashboard(args):
    app_path = "src/thucia/viz/dashboard/app.py"
    subprocess.run(["streamlit", "run", app_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="Thucia")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dash = subparsers.add_parser("dashboard", help="Start the Thucia dashboard server")
    dash.set_defaults(func=launch_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
