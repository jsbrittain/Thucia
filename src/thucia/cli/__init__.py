from dataclasses import dataclass
from typing import List

import thucia.cli.steps as steps


class Command:
    """Base class for commands. Subclasses implement `run(args)`."""

    name: str = "base"
    help: str = "base command"
    arguments: List[dict] = []

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


CommandsList = [
    DashboardCommand,
    CasesPerMonth,
    PlotCasesPerMonth,
    CasesPerWeek,
    PlotCasesPerWeek,
    CasesPerDay,
    PlotCasesPerDay,
]
