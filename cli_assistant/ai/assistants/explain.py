from rich.console import Console
from rich.markdown import Markdown

from ..agent import Agent, Environment


SYSTEM_PROMPT = """
You are a highly experienced Unix system administrator and command line expert. Given a Bash command, 
output a detailed explanation of what the command does, including its components and their roles.
Also provide one or 2 examples on how to use that command in a real-world scenario.
If the input provided is a complex command (e.g., a pipeline), break it down into its components
and explain each part.

Use simple language that is easy to understand and format the output using Markdown for readability.
"""

def explain(config: dict, command: str):
    environment = Environment()
    agent = Agent(config, environment, SYSTEM_PROMPT)

    user_task = f"Explain the following command: '{command}'"
    response = agent.run(user_task, max_iterations=1)
    description = response.content or "The AI failed to generate a description."

    console = Console()
    markdown = Markdown(description)
    console.print(markdown)
