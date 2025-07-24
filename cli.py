#!/usr/bin/env python3

import argparse
import argcomplete
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Optional

_available_commands: List["Command"] = []


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
    #TODO
    pass


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
    """Just chat with an AI from the command line."""

    if args.list and args.resume:
        print(
            "Error: Cannot use --list and --resume at the same time.", file=sys.stderr
        )
        sys.exit(1)

    if args.list:
        print("TODO: List previous chat sessions.")
    elif args.resume:
        print(f"Resuming '{args.resume}'")
    else:
        print("TODO: Start a new interactive chat session.")


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
    print(f"TODO: Send this prompt to an AI service: '{args.cmd}'")


@command(
    [
        PositionalArg(
            name="prompt",
            help="The natural language text to translate into a shell command.",
        )
    ]
)
def handle_do(args):
    """Describe in plain English the task you want to perform."""
    print(f"TODO: Send this prompt to an AI service: '{args.prompt}'")


@command(
    [
        PositionalArg(
            name="path",
            help="The path to the file or directory to summarize.",
        )
    ]
)
def handle_summarize(args):
    """Summarizes the content of a given file or directory."""
    print(f"TODO: Summarize the content of '{args.path}'")


##############################################################################


def main():
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

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
