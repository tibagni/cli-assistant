import json
import sys
import os

from typing import Dict, Optional
from dataclasses import dataclass

from ..agent import Agent, Environment

SYSTEM_PROMPT = """
You are a highly experienced Unix system administrator and command line expert. Given a natural 
language task, output a Bash command that solves it, along with a safety assessment and explanation.

Your response must contain the following information:
1. A bash command that solves the problem. If the problem can't be solved, return an empty string 
    for the command 
2. A risk assessment of the command, as follows:
    - Risk 0: Safe to run. No data is modified or deleted.
    - Risk 1: May modify or delete user data, but not system-critical files.
    - Risk 2: May impact system integrity, security, or availability (e.g., running as root, 
        altering system files).
    - If the command is empty, the risk assessment must also be 0.
3. An explanation of the command and how it achieves the solution.
4. A disclaimer message shown to the user if the command carries any risk (i.e., risk > 0). This
message must warn the user about what could happen if the command is executed. If the command is
safe (risk 0), return an empty string.

Make sure your response is a valid JSON object in the following format (with double quotes and no 
trailing commas):
{
    "command": "<bash command>",
    "risk_assessment": <0|1|2>,
    "explanation": "<explanation of the command>",
    "disclaimer": "<disclaimer shown if risk > 0, empty string otherwise>"
}

Ensure the JSON is valid and can be parsed with standard JSON parsers. 
Do not include markdown, code blocks, or comments. Only output the JSON.
"""

@dataclass
class CommandSuggestion:
    """Represents a command suggestion from the AI, including metadata."""

    command: str
    risk_assessment: int
    explanation: str
    disclaimer: str


def _suggest_shell_command(config: Dict, natual_language_description: str) -> Optional[CommandSuggestion]:
    # For single-shot tasks, we can use a default, non-interactive environment.
    environment = Environment()
    agent = Agent(config, environment, SYSTEM_PROMPT)

    # The agent runs for one cycle and returns the final response.
    response = agent.run(natual_language_description, max_iterations=1)

    if response.content:
        try:
            data = json.loads(response.content)
            return CommandSuggestion(**data)
        except (json.JSONDecodeError, TypeError):
            return CommandSuggestion(
                command="",
                risk_assessment=0,
                explanation="Error: The AI failed to return a valid command.",
                disclaimer="",
            )

    return None

def do(config: Dict, natual_language_description: str):
    result = _suggest_shell_command(config, natual_language_description)
    if not result or not result.command:
        print("Could not generate a command for the given prompt.", file=sys.stderr)
        sys.exit(1)

    print(f"Suggested command:\n  {result.command}\n")
    print(f"Explanation:\n  {result.explanation}\n")

    if result.risk_assessment > 0 and result.disclaimer:
        print(f"⚠️  Disclaimer:\n  {result.disclaimer}\n")

    try:
        confirm = input("Do you want to run this command? [y/N] ")
        if confirm.lower() == "y":
            os.system(result.command)
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)