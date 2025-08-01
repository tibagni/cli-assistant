import os
from typing import Dict, Optional

from rich.console import Console
from rich.markdown import Markdown

from ..agent import Agent, Environment

SYSTEM_PROMPT = """
You are a software architect and developer assistant. Your goal is to help the user
scaffold a new project by creating directories and files based on their request.

First, think step-by-step about the file structure and content needed to fulfill
the user's request. Then, call the available tools sequentially to create the
project. Do not try to create a file inside a directory that hasn't been created yet.

Once you have created all the necessary files and directories, respond with a
final confirmation message summarizing what you have done.
"""


class BoilerplateEnvironment(Environment):
    """Provide the necessary tools for file and directory creation for boilerplate projects."""

    def __init__(self):
        super().__init__()
        self.console = Console()

    @Environment.tool()
    def create_directory(self, path: str):
        """
        Creates a directory at the specified path.
        Fails if the directory already exists or if the parent does not exist.
        """
        if os.path.exists(path):
            return f"Error: Path '{path}' already exists."
        try:
            os.makedirs(path)
            self.console.print(f"[green]✓ Created directory:[/] {path}")
            return f"Successfully created directory: {path}"
        except Exception as e:
            return f"Error creating directory '{path}': {e}"

    @Environment.tool()
    def create_file(self, path: str, content: str):
        """
        Creates a file at the specified path with the given content.
        Fails if the file already exists.
        """
        if os.path.exists(path):
            return f"Error: File '{path}' already exists."
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.console.print(f"[green]✓ Wrote file:[/] {path}")
            return f"Successfully created file: {path}"
        except Exception as e:
            return f"Error creating file '{path}': {e}"

def _do_boilerplate(config: Dict, description: str):
    environment = BoilerplateEnvironment()
    agent = Agent(config, environment, SYSTEM_PROMPT)

    # Allow the agent to run for multiple iterations to call tools
    return agent.run(description, max_iterations=10)


def boilerplate(config: Dict, description: str):
    """Generates project boilerplate from a natural language description."""
    console = Console()
    console.print("[bold yellow]Starting boilerplate generation...[/]")

    response = _do_boilerplate(config, description)

    console.print("\n[bold green]Boilerplate generation complete![/]")
    if response.content:
        markdown = Markdown(response.content)
        console.print(markdown)