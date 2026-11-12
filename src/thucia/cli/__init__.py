from dataclasses import dataclass
from typing import List

import thucia.cli.steps as steps


class Command:
    """Base class for commands. Subclasses implement `run(args)`."""

    name: str = "base"
    help: str = "base command"
    arguments: List[dict] = []
    #: Extra argv a command needs to parse at all (e.g. a required positional),
    #: used by the parser-wiring tests.
    minimal_argv: tuple = ()

    @classmethod
    def add_to_subparsers(cls, subparsers):
        p = subparsers.add_parser(cls.name, help=cls.help, description=cls.help)
        for arg in cls.arguments:
            flags = arg.get("flags", ())
            kwargs = dict(arg)
            kwargs.pop("flags", None)
            kwargs.pop("positional", None)
            p.add_argument(*flags, **kwargs)
        p.set_defaults(_command_class=cls)

    def execute(self, cmd_args):
        cmd_args = {k: v for k, v in vars(cmd_args).items() if not k.startswith("_")}
        # Extract positional and keyword arguments
        args = []
        kwargs = {}
        flags = [
            flag
            for arg in self.arguments
            for flags in arg.get("flags", ())
            for flag in flags
        ]
        for name, value in cmd_args.items():
            if name in flags:
                args.append(value)
            else:
                kwargs[name] = value
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class DashboardCommand(Command):
    name = "dashboard"
    help = "Start the Thucia dashboard server."

    def run(self, *args, **kwargs):
        return steps.launch_dashboard()


@dataclass
class CasesPerMonth(Command):
    name = "cases-per-month"
    help = "Aggregate case counts per month and write result."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name containing case counts.",
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--output-file",),
            "type": str,
            "default": "cases_per_month",
            "positional": False,
            "help": "Output DB/file label (passed to write_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.cases_per_month(*args, **kwargs)


@dataclass
class PlotCasesPerMonth(Command):
    name = "plot-cases-per-month"
    help = "Plot case counts per month."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases_per_month",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
        {
            "flags": ("--title",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Title for the plot.",
        },
        {
            "flags": ("--date-col",),
            "type": str,
            "default": "Date",
            "positional": False,
            "help": "Column name for dates.",
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name for case counts.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.plot_cases_per_month(*args, **kwargs)


class CasesPerWeek(Command):
    name = "cases-per-week"
    help = "Aggregate case counts per week and write result."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name containing case counts.",
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--output-file",),
            "type": str,
            "default": "cases_per_week",
            "positional": False,
            "help": "Output DB/file label (passed to write_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.cases_per_week(*args, **kwargs)


class PlotCasesPerWeek(Command):
    name = "plot-cases-per-week"
    help = "Plot case counts per week."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases_per_week",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
        {
            "flags": ("--title",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Title for the plot.",
        },
        {
            "flags": ("--date-col",),
            "type": str,
            "default": "Date",
            "positional": False,
            "help": "Column name for dates.",
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name for case counts.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.plot_cases_per_week(*args, **kwargs)


class CasesPerDay(Command):
    name = "cases-per-day"
    help = "Aggregate case counts per day and write result."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name containing case counts.",
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--output-file",),
            "type": str,
            "default": "cases_per_day",
            "positional": False,
            "help": "Output DB/file label (passed to write_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.cases_per_day(*args, **kwargs)


class PlotCasesPerDay(Command):
    name = "plot-cases-per-day"
    help = "Plot case counts per day."

    arguments = [
        {
            "flags": ("--project",),
            "type": str,
            "default": "cases",
            "positional": False,
            "help": (
                "Project name (subfolder under projects root). "
                "Use empty string to use current directory."
            ),
        },
        {
            "flags": ("--cases-file",),
            "type": str,
            "default": "cases_per_day",
            "positional": False,
            "help": "Input DB/file label (passed to read_db).",
        },
        {
            "flags": ("--projects-root",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Optional override for projects root path.",
        },
        {
            "flags": ("--title",),
            "type": str,
            "default": None,
            "positional": False,
            "help": "Title for the plot.",
        },
        {
            "flags": ("--date-col",),
            "type": str,
            "default": "Date",
            "positional": False,
            "help": "Column name for dates.",
        },
        {
            "flags": ("--cases-col",),
            "type": str,
            "default": "Cases",
            "positional": False,
            "help": "Column name for case counts.",
        },
    ]

    def run(self, *args, **kwargs):
        steps.plot_cases_per_day(*args, **kwargs)


@dataclass
class ProvenanceCommand(Command):
    name = "provenance"
    help = "Show (or verify) the run metadata recorded in a saved artefact."

    minimal_argv = ("run.duckdb",)

    arguments = [
        {
            "flags": ("file",),
            "type": str,
            "help": (
                "Saved artefact to inspect: a .duckdb/.nc/.zarr written by "
                "fit_model (or any file carrying __run_metadata__)."
            ),
        },
        {
            "flags": ("--document",),
            "type": str,
            "default": None,
            "help": (
                "Print only this document (selection, config, code, inputs, "
                "environment). Default: all documents."
            ),
        },
        {
            "flags": ("--json",),
            "dest": "as_json",
            "action": "store_true",
            "help": "Emit compact JSON instead of indented JSON.",
        },
        {
            "flags": ("--verify",),
            "action": "store_true",
            "help": (
                "Compare the recorded implementation with the one running now; "
                "exit 1 when they differ."
            ),
        },
        {
            "flags": ("--include-deps",),
            "dest": "include_deps",
            "action": "store_true",
            "help": (
                "With --verify, also compare the numerics-stack versions "
                "(darts/torch/statsmodels/numpy/pandas/duckdb)."
            ),
        },
    ]

    def run(self, file, *args, **kwargs):
        return steps.provenance(file, **kwargs)


CommandsList = [
    DashboardCommand,
    CasesPerMonth,
    PlotCasesPerMonth,
    CasesPerWeek,
    PlotCasesPerWeek,
    CasesPerDay,
    PlotCasesPerDay,
    ProvenanceCommand,
]
