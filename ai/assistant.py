import json
from typing import Dict, Optional

from .agent import Agent, Environment


def get_shell_command(config: Dict, natual_language_description: str) -> Optional[Dict]:
    prompt = """
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

    # For single-shot tasks, we can use a default, non-interactive environment.
    environment = Environment()
    agent = Agent(config, environment, prompt)

    # The agent runs for one cycle and returns the final response.
    response = agent.run(natual_language_description, max_iterations=1)

    if response.content:
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # If the LLM fails to return valid JSON, create a default error response.
            return {
                "command": "",
                "risk_assessment": 0,
                "explanation": "Error: The AI failed to return a valid command.",
                "disclaimer": "",
            }

    return None