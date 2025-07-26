import json
import sys
import os

from typing import Dict, Optional
from dataclasses import dataclass

from ..agent import Agent, Environment

COMMAND_SUGGESTION_SCHEMA = {
    "name": "command_suggestion",
    "schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The suggested bash command. Can be an empty string if no command is suitable.",
            },
            "risk_assessment": {
                "type": "integer",
                "description": "A numerical assessment of the command's potential risk.",
                "enum": [0, 1, 2],
            },
            "explanation": {
                "type": "string",
                "description": "A detailed, human-readable explanation of what the command does and how it works.",
            },
            "disclaimer": {
                "type": "string",
                "description": "A warning message to be displayed for commands with a risk level greater than 0. Should be an empty string for risk 0.",
            },
        },
        "required": ["command", "risk_assessment", "explanation", "disclaimer"],
    }
}

SYSTEM_PROMPT = """
You are a highly experienced Unix system administrator and command line expert.
Given a natural language task, your goal is to provide a safe and effective Bash command.

Your response must be a JSON object that conforms to the provided schema.

The core requirements for your response are:
1.  **command**: The corresponding Bash command. If no command is suitable, provide an empty string.
2.  **risk_assessment**: A numerical rating of the command's potential risk:
    - 0: Safe to run (e.g., read-only operations like `ls`, `cat`, `grep`).
    - 1: Potentially destructive (e.g., modifies or deletes user files like `mv`, `cp`, `rm`).
    - 2: High risk (e.g., requires `sudo`, alters system files, affects security/availability).
3.  **explanation**: A clear, concise explanation of what the command does.
4.  **disclaimer**: A warning about potential consequences if the risk is 1 or 2. This should be an empty string for risk 0.
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
    # By providing the schema, we instruct the LLM to return a JSON object
    # that matches the specified structure, making the output more reliable.
    response = agent.run(
        natual_language_description,
        max_iterations=1,
        response_format={"type": "json_schema", "json_schema": COMMAND_SUGGESTION_SCHEMA},
    )

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