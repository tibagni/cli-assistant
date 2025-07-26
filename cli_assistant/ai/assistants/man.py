import subprocess

from typing import Dict
from rich.console import Console
from rich.markdown import Markdown

from ..agent import Agent, Environment


SYSTEM_PROMPT = """
You are a helpful AI assistant. The user will provide the name of a Unix command.
Your job is to read the man page (provided as context) for that command and produce:
1. A concise summary of what the command does.
2. The most useful/common options (with a brief description for each).
3. 2-3 practical usage examples.
Respond in a clear, beginner-friendly format using Markdown.
If the man page is very long, focus only on the most important highlights.
Don't include any follow up options. Don't ask if the user needs more clarifications or has any questions.
"""

def _fetch_man_page(command: str) -> str:
    try:
        result = subprocess.run(
            ["man", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception:
        return ""

def _generate_man_summary(config: Dict, command: str) -> str:
    environment = Environment()
    agent = Agent(config, environment, SYSTEM_PROMPT)

    man_page = _fetch_man_page(command)
    if not man_page:
        return f"Could not find a man page for `{command}`."

    user_task = f"Here is the man page for '{command}':\n\n{man_page}\n\nSummarize it as described, using Markdown for formatting."
    response = agent.run(user_task, max_iterations=1)
    return response.content or "The AI failed to generate a summary."

def man(config: Dict, command: str):
    """
    Fetches and displays a summarized, easy-to-read version of a man page.
    """
    summary_markdown = _generate_man_summary(config, command)

    console = Console()
    markdown = Markdown(summary_markdown)
    console.print(markdown)