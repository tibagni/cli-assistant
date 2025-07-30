import os
import subprocess
from typing import Dict, Optional

from rich.console import Console
from rich.markdown import Markdown

from cli_assistant.ai.llm import LLMCompletionResponse

from ..agent import Agent, Environment
from .summarize import _do_summarize

SYSTEM_PROMPT = """
You are a general-purpose AI assistant running in a command-line environment.
You can help with a wide range of topics — from writing, coding, and troubleshooting
to general knowledge — but you are especially skilled at working with command-line tools,
Bash scripting, and Unix-based systems. You can access special tools that allow you to interact
with the user's environment

Not all tools will succeed — some actions may be denied or blocked by the user — 
so you must be resilient and able to continue the conversation even when a tool call fails.

Always be concise and clear, especially when helping with technical or Bash-related tasks.
Explain commands when useful, and warn about potentially destructive operations.
Assume you're working in a Unix-like terminal unless told otherwise.

While you're optimized for the command line, you can help with anything the user
might ask — from explaining concepts to writing code, reviewing documents, or offering
advice — just like ChatGPT.

You always aim to be helpful, safe, and user-respecting.
"""

FILE_SIZE_LIMIT = 40000

class ChatEnvironment(Environment):
    def __init__(self, ai_config: Dict):
        super().__init__()
        self.ai_config = ai_config
        self.console = Console()

    @Environment.tool()
    def get_current_working_directory(self) -> str:
        """Returns the current working directory."""
        cwd = os.getcwd()
        self.console.print(f"[green]✓ querying cwd:[/] {cwd}")
        return cwd

    @Environment.tool()
    def list_files_and_dirs(self, path: str = ".") -> str:
        """Lists all files and directories in a given path. If no path is provided,
        defaults to the current directory."""

        try:
            if not os.path.isdir(path):
                return f"Error: '{path}' is not a valid directory."

            files = [
                f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            ]
            folders = [
                f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
            ]

            self.console.print(f"[green]✓ Listing files in:[/] {path}")
            return f"Files: {', '.join(files)}\nFolders: {', '.join(folders)}"
        except Exception as e:
            return f"Error listing files in '{path}': {e}"

    @Environment.tool()
    def summarize_path(self, path: str):
        """
        Provides a quick summary of a file or an overview of a directory's contents.
        This is useful for getting a high-level understanding without reading the full content.
        """
        # This reuses the helper from summarize.py
        self.console.print(f"[green]✓ Summarizing:[/] {path}")
        return _do_summarize(self.ai_config, [path]).content

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
        
    @Environment.tool()
    def read_file(self, path: str):
        """Reads the entire content of a specified file. Limits to 8000 characters."""
        try:
            if not os.path.isfile(path):
                return f"Error: '{path}' is not a file."
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                # Add a limit to avoid huge files blowing up the context
                content = f.read(FILE_SIZE_LIMIT)
                if len(content) == FILE_SIZE_LIMIT:
                    content += "\n... (file content truncated)"

                self.console.print(f"[green]✓ Reading file:[/] {path}")                    
                return content
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    @Environment.tool()
    def run_command(self, command: str) -> str:
        """
        Runs a shell command and returns its standard output and standard error.
        Use this tool to execute any command line utility.
        """

        # Ask the user if it is OK to run the command first
        if not self._get_user_confirmation(f"Run command '{command}'?"):
            return "Aborted by user. Command not executed."

        self.console.print(f"[green]✓ Running command:[/] {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,  # Add a timeout to prevent hanging
            )
            output = result.stdout.strip()
            error = result.stderr.strip()
            if output:
                return f"STDOUT:\n{output}"
            if error:
                return f"STDERR:\n{error}"
            return "Command executed successfully with no output."
        except subprocess.CalledProcessError as e:
            return f"Command failed with exit code {e.returncode}.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        except FileNotFoundError:
            return f"Error: Command not found. Make sure '{command.split()[0]}' is in your PATH."
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after 60 seconds."
        except Exception as e:
            return f"An unexpected error occurred: {e}"
        
    def _get_user_confirmation(self, message: str) -> bool:
        try:
            confirm = input(f"{message} [y/N] ")
            return confirm.lower() == "y"
        except (KeyboardInterrupt, EOFError):
            return False
    
    # Keep the agent interactive
    def handle_agent_response(self, agent_response: LLMCompletionResponse) -> Optional[str]:
        response = agent_response.content
        if not response:
            return None
        

        markdown = Markdown(response)
        self.console.print(markdown)

        user_response = ""
        try:
            while not user_response:
                user_response = input("> ") 
            return user_response
        except (KeyboardInterrupt, EOFError):
            return None


def chat(config: Dict):
    """Starts an interactive chat with the AI."""
    environment = ChatEnvironment(config)
    agent = Agent(config, environment, SYSTEM_PROMPT)

    user_prompt = input("Ask anything: ")

    # Just call the agent with a huge limit of iterations and et the user talk to it
    agent.run(user_prompt, max_iterations=10000)
