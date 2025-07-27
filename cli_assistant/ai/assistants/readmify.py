import os
import subprocess
from typing import Dict, List

from rich.console import Console

from ..agent import Agent, Environment

# Re-using the file reading logic from the summarize assistant is a pragmatic
# way to avoid code duplication. A future refactor could move these helpers
# to a shared utility module.
from .summarize import _do_summarize

SYSTEM_PROMPT = """
You are an expert technical writer specializing in creating high-quality README.md files for software projects.
Your goal is to understand the project structure and content by exploring it with the available tools, and then generate a comprehensive README.md.

Make use of the tools available to you to accomplish this task

Your process should be:
1. Start by listing the files in the root directory (`.`) to get an overview.
2. Use `summarize_path` and `read_file` on key files to understand the project's purpose, dependencies, and main logic.
3. Use `get_git_history` to understand recent changes.
4. Once you have a good understanding, call the `write_readme` tool with the full, well-structured markdown content.
"""


class ReadmifyEnvironment(Environment):
    """An environment with tools for project exploration and README writing."""

    def __init__(self, path: str, ai_config: Dict):
        super().__init__()
        self.path = path
        self.console = Console()
        # Change working directory so file operations are relative to the project path
        self.original_cwd = os.getcwd()
        os.chdir(self.path)
        self.ai_config = ai_config

    def __del__(self):
        # Restore original working directory when the object is destroyed
        os.chdir(self.original_cwd)

    @Environment.tool()
    def list_files_and_dirs(self, path: str = "."):
        """Lists all files and directories in a given path. If no path is provided, defaults to the current directory."""

        try:
            if not os.path.isdir(path):
                return f"Error: '{path}' is not a valid directory."
            
            self.console.print(f"[green]✓ Listing files in:[/] {path}")
            return "\n".join(os.listdir(path))
        except Exception as e:
            return f"Error listing files in '{path}': {e}"

    @Environment.tool()
    def read_file(self, path: str):
        """Reads the entire content of a specified file."""
        try:
            if not os.path.isfile(path):
                return f"Error: '{path}' is not a file."
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                # Add a limit to avoid huge files blowing up the context
                content = f.read(8000)
                if len(content) == 8000:
                    content += "\n... (file content truncated)"

                self.console.print(f"[green]✓ Reading file:[/] {path}")                    
                return content
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    @Environment.tool()
    def summarize_path(self, path: str):
        """
        Provides a quick summary of a file or an overview of a directory's contents.
        This is useful for getting a high-level understanding without reading the full content.
        """
        # This reuses the helper from summarize.py, which is a good pattern.
        self.console.print(f"[green]✓ Summarizing:[/] {path}")
        return _do_summarize(self.ai_config, [path]).content

    @Environment.tool()
    def get_git_history(self, limit: int = 10):
        """
        Retrieves the most recent git commit history for the project.
        Returns the last `limit` commits.
        """
        try:
            result = subprocess.run(
                ["git", "log", f"-n{limit}", "--pretty=format:%h - %an, %ar : %s"],
                capture_output=True, text=True, check=True
            )

            self.console.print(f"[green]✓ Getting git history:[/] last {limit} commits")
            return result.stdout
        except FileNotFoundError:
            return "Error: git command not found. Is git installed and in your PATH?"
        except subprocess.CalledProcessError:
            return "Error: This does not appear to be a git repository."
        except Exception as e:
            return f"An unexpected error occurred while reading git history: {e}"

    @Environment.tool()
    def get_git_commit(self, commit_hash: str):
        """
        Retrieves the details and changes for a specific git commit hash.
        """
        try:
            result = subprocess.run(
                ["git", "show", commit_hash],
                capture_output=True, text=True, check=True
            )

            self.console.print(f"[green]✓ Reading commit:[/] {commit_hash}")
            return result.stdout
        except FileNotFoundError:
            return "Error: git command not found. Is git installed and in your PATH?"
        except subprocess.CalledProcessError:
            return f"Error: Could not find commit hash '{commit_hash}'."
        except Exception as e:
            return f"An unexpected error occurred while reading git commit: {e}"

    @Environment.tool()
    def write_readme(self, content: str):
        """
        Writes the provided markdown content to a README.md file in the project directory.
        If a README.md already exists, it will ask for confirmation before overwriting.
        """
        readme_path = "README.md" # Relative to the new cwd
        if os.path.exists(readme_path):
            # Restore cwd for input, then change back
            os.chdir(self.original_cwd)
            confirm = input(f"'{os.path.join(self.path, readme_path)}' already exists. Overwrite? [y/N] ")
            os.chdir(self.path)
            if confirm.lower() != "y":
                return "Aborted by user. README.md was not overwritten."

        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.console.print(f"[green]✓ Successfully wrote README.md to:[/] {os.path.join(self.path, readme_path)}")
            return f"Successfully wrote README.md to {os.path.join(self.path, readme_path)}"
        except Exception as e:
            return f"Error writing README.md: {e}"


def readmify(config: Dict, path: str):
    """Generates a README.md file for a given project path."""
    console = Console()
    console.print(f"[bold yellow]Analyzing project at '{path}' to generate README...[/]")

    # The agent will now do the exploration. We just need to give it the starting path.
    user_task = f"Please generate a README.md file for the project located at '{path}'. Start by exploring the project using the available tools."

    environment = ReadmifyEnvironment(path, config)
    agent = Agent(config, environment, SYSTEM_PROMPT)

    # Allow more iterations for exploration
    response = agent.run(user_task, max_iterations=25)
    if response.interrupted:
        console.print("\n[bold yellow]Warning: Max iterations reached before the agent could finish.[/]")
    else:
        console.print("\n[bold green]README generation process complete![/]")
