import inspect
import os
import json
import readline

from .llm import LLMClient, LLMCompletionResponse
from typing import List, Dict, Optional, get_type_hints


class Environment:
    def __init__(self):
        self.tools = {}
        self._collect_tools()

    def _collect_tools(self):
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)
            if hasattr(attr, "__tool_info__"):
                self.tools[attr_name] = getattr(attr, "__tool_info__")

    def get_tools(self) -> List[Dict]:
        # Format the function in the OpenAI format
        return [
            {
                "type": "function",
                "function": {
                    "name": t["tool_name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self.tools.values()
        ]

    def run_tool(self, tool_name: str, args: Dict) -> str:
        tool_info = self.tools.get(tool_name)
        if tool_info:
            func = tool_info["function"]
            return func(self, **args)
        else:
            raise ValueError(f"Tool '{tool_name}' not found.")

    def read_user_prompt(self, message: str = "> ") -> str:
        return input(f"{message} ")

    @staticmethod
    def tool():
        def decorator(func):
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)

            # Build JSON schema for arguments
            args_schema = {"type": "object", "properties": {}, "required": []}

            param_types = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            # Examine each parameter
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                # Convert Python types to JSON schema types
                param_type = type_hints.get(param_name, str)

                param_schema = {"type": param_types.get(param_type, "string")}

                args_schema["properties"][param_name] = param_schema

                # If parameter has no default, it's required
                if param.default == inspect.Parameter.empty:
                    args_schema["required"].append(param_name)

            tool_description = func.__doc__.strip() if func.__doc__ else ""
            func.__tool_info__ = {
                "function": func,
                "tool_name": func.__name__,
                "description": tool_description,
                "parameters": args_schema,
            }
            return func

        return decorator
    
    def handle_agent_response(self, agent_response: LLMCompletionResponse) -> Optional[str]:
        # By default, let the agent finish without further user prompts
        return None


class Agent:
    def __init__(self, config: Dict, env: Environment, system_prompt: str = ""):
        self.config = config
        self.env = env
        self.llm = LLMClient(self.config["provider_configs"])
        self._messages = []
        if system_prompt:
            self._messages.append(LLMClient.format_system_message(system_prompt))

    def run(
        self, user_task: str, max_iterations: int = 10
    ) -> LLMCompletionResponse:
        """
        Executes a task with the agent, returning the final completion response.
        """
        self._messages.append(LLMClient.format_user_message(user_task))

        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        # This variable will be updated in each loop iteration and is guaranteed
        # to be set because max_iterations is positive.
        response: LLMCompletionResponse = LLMCompletionResponse({})

        for _ in range(max_iterations):
            response = self.llm.completion(
                model=f"{self.config["provider"]}:{self.config['model']}",
                messages=self._messages,
                tools=self.env.get_tools(),
                max_tokens=1024,
            )

            # The assistant's response (including content and any tool calls)
            # must be added to the history.
            self._messages.append(response.assistant_message)

            if response.tool_calls:
                # For each tool call, execute it and append the result.
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    tool_id = tool_call["id"]
                    try:
                        result = self.env.run_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    self._messages.append(
                        LLMClient.format_tool_message(str(result), tool_id)
                    )
            else:
                # No more actions, let the environment know the gent response
                # and provide further instructions if needed.
                new_instructions = self.env.handle_agent_response(response)
                if new_instructions:
                    self._messages.append(LLMClient.format_user_message(new_instructions))
                else:
                    # The agent's turn is over.
                    return response

        # This is reached if max_iterations is hit.
        return response


# ===================================================================================


class MyEnvironment(Environment):
    @Environment.tool()
    def list_files(self, path: str = os.getcwd()) -> List[str]:
        """List files in the given path. If no path is given, it will use the current path."""
        return os.listdir(path)

    @Environment.tool()
    def read_file(self, file_path: str) -> str:
        """Read the content of the file specified in 'file_path'."""
        with open(file_path, "r") as file:
            return file.read()


class InteractiveEnvironment(MyEnvironment):
    """An environment that prompts the user for input when the agent finishes."""

    def handle_agent_response(
        self, agent_response: LLMCompletionResponse
    ) -> Optional[str]:
        """After the agent responds, print its message and prompt for the next task."""
        if agent_response.content:
            # In an interactive session, we print the agent's final message.
            print(f"\n{agent_response.content}\n")
        return self.read_user_prompt()


def main():
    # Construct a robust path to config.json relative to this script's location
    # to ensure the test runner works correctly when run as a module.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.json")

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    environment = InteractiveEnvironment()
    task = environment.read_user_prompt("What do you want to do: ")
    agent = Agent(config, environment)
    agent.run(task, max_iterations=200)


def main_test():
    env = MyEnvironment()
    print(env.tools)


if __name__ == "__main__":
    main()
    # main_test()
