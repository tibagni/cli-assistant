#!/usr/bin/env python3

import argparse
import argcomplete
import sys
import os
import json

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional
from .ai import do, explain, man, summarize, boilerplate, readmify, chat


_available_commands: List["Command"] = []
_ai_config: Dict = {}


@dataclass
class Argument(ABC):
    def __init__(self, help: str, kwargs: Optional[dict] = None):
        self.help = help
        self.kwargs = kwargs if kwargs is not None else {}

    @abstractmethod
    def add_to_parser(self, parser: argparse.ArgumentParser):
        pass


class OptionalArg(Argument):
    def __init__(
        self,
        short_option: str,
        long_option: str,
        help: str,
        kwargs: Optional[dict] = None,
    ):
        super().__init__(help=help, kwargs=kwargs)
        self.short_option = short_option
        self.long_option = long_option

    def add_to_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            self.short_option, self.long_option, help=self.help, **self.kwargs
        )


class PositionalArg(Argument):
    def __init__(self, name: str, help: str, kwargs: Optional[dict] = None):
        super().__init__(help=help, kwargs=kwargs)
        self.name = name

    def add_to_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(self.name, help=self.help, **self.kwargs)


@dataclass
class Command:
    name: str
    func: Callable
    help: str
    description: str
    args: list[Argument]


def _validate_ai_config():
    global _ai_config
    if not _ai_config:
        # TODO read it from ~/.cli-assist in the future
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        # TODO ask the user to create the config file if it does not exist
        # Create a default template and ask the user if they want to open with $EDITOR. If not
        # let them choose the editor, then open the template so they can edit the configs

        with open(config_path, "r") as config_file:
            _ai_config = json.load(config_file)


def command(args: List[Argument]):
    def decorator(func):
        if not func.__name__.startswith("handle_"):
            raise ValueError("Command handler must start with 'handle_'.")

        if not func.__doc__:
            raise ValueError(
                f"Command handler '{func.__name__}' must have a docstring for its help text."
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            _validate_ai_config()
            return func(*args, **kwargs)

        command_name = func.__name__.split("_")[1]
        # Use the first line of the docstring as the help text and
        # the full docstring for the detailed description.
        help_text = func.__doc__.strip().split("\n")[0]
        _available_commands.append(
            Command(command_name, wrapper, help_text, func.__doc__, args)
        )
        return wrapper

    return decorator


##############################################################################


@command(
    [
        OptionalArg(
            short_option="-l",
            long_option="--list",
            help="List any previous chat sessions.",
            kwargs={"action": "store_true"},
        ),
        OptionalArg(
            short_option="-r",
            long_option="--resume",
            help="Resume the provided chat session.",
        ),
    ]
)
def handle_chat(args):
    """Just chat with an AI from the command line.
    The AI will have acces to your local shell and can execute commands (if you allow). Go nuts
    """
    # TODO handle args
    chat(_ai_config)


@command(
    [
        PositionalArg(
            name="cmd",
            help="The shell command you wish to understand more of.",
        )
    ]
)
def handle_explain(args):
    """Get a detailed explanation of any given shell command."""
    explain(_ai_config, args.cmd)


@command(
    [
        PositionalArg(
            name="prompt",
            help="The natural language text to translate into a shell command.",
        )
    ]
)
def handle_do(args):
    """Run a shell command based on a natural language description."""
    do(_ai_config, args.prompt)


@command(
    [
        PositionalArg(
            name="paths",
            help="One or more paths to files or directories to summarize.",
            kwargs={"nargs": "+"},
        )
    ]
)
def handle_summarize(args):
    """Summarizes the content of a given file or directory."""
    summarize(_ai_config, args.paths)


@command(
    [
        PositionalArg(
            name="page",
            help="The man page you want explained in simple terms",
        )
    ]
)
def handle_man(args):
    """Summarizes and explains in simple terms, with examples, the contents of a man page."""
    man(_ai_config, args.page)


@command(
    [
        PositionalArg(
            name="description",
            help="A description of the project boilerplate to generate.",
        )
    ]
)
def handle_boilerplate(args):
    """Generates project boilerplate from a description."""
    boilerplate(_ai_config, args.description)


@command(
    [
        PositionalArg(
            name="path",
            help="The path to the project directory to document. Defaults to the current directory.",
            kwargs={"nargs": "?", "default": "."},
        )
    ]
)
def handle_readmify(args):
    """Generates a README.md file for a project directory."""
    readmify(_ai_config, args.path)


##############################################################################


def run_cli(argv: Optional[List[str]] = None):
    """
    Parses command-line arguments and executes the corresponding command.

    This function is designed to be testable by allowing arguments to be passed
    directly.

    Args:
        argv: A list of strings representing the command-line arguments.
              If None, `sys.argv[1:]` is used automatically by `parse_args`.
    """
    parser = argparse.ArgumentParser(
        description="An AI-powered command-line assistant to supercharge your terminal."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Sub-commands", required=True
    )

    # Sort commands alphabetically for consistent --help output.
    _available_commands.sort(key=lambda cmd: cmd.name)

    for command in _available_commands:
        subparser = subparsers.add_parser(
            command.name, help=command.help, description=command.description
        )
        for arg in command.args:
            arg.add_to_parser(subparser)
        subparser.set_defaults(func=command.func)

    # Enable argument auto-completion.
    argcomplete.autocomplete(parser)

    # When `argv` is None (the default), `parse_args` automatically uses `sys.argv[1:]`.
    # This is the standard behavior for a real command-line application.
    # When a list is provided (as in our tests), `parse_args` uses that list directly.
    # This allows us to test the parser without mocking `sys.argv`.
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """The main entry point for the command-line interface, called by the `assist` script."""
    # This function is the script's entry point. It calls the core logic
    # without arguments, ensuring it uses the actual command-line arguments from `sys.argv`.
    run_cli()


if __name__ == "__main__":
    main()
