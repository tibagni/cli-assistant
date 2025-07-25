from dataclasses import dataclass
import aisuite

from typing import Dict, List, Optional


@dataclass
class LLMCompletionResponse:
    """Wraps the full assistant message from the LLM API."""

    assistant_message: Dict

    @property
    def content(self) -> Optional[str]:
        """The text content of the message, if any."""
        return self.assistant_message.get("content")

    @property
    def tool_calls(self) -> Optional[List[Dict]]:
        """The list of tool calls requested by the LLM, if any."""
        return self.assistant_message.get("tool_calls")


class LLMClient:
    """
    A wrapper for the LLM client to abstract away the specific provider library.
    This allows for easier swapping of LLM providers in the future.
    """

    def __init__(self, provider_configs: Dict):
        """
        Initializes the LLM client.

        Args:
            provider_configs: A dictionary containing configuration for the LLM provider.
        """
        self.client = aisuite.Client(provider_configs)

    @staticmethod
    def format_system_message(content: str) -> Dict:
        return {"role": "system", "content": content}

    @staticmethod
    def format_user_message(content: str) -> Dict:
        return {"role": "user", "content": content}

    @staticmethod
    def format_assistant_message(content: str) -> Dict:
        return {"role": "assistant", "content": content}

    @staticmethod
    def format_tool_message(content: str, tool_call_id: str) -> Dict:
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id}

    def completion(
        self,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMCompletionResponse:
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=tools, **kwargs
        )

        # The message object from aisuite/openai can be converted to a dict.
        # We exclude unset values to keep the payload clean and compatible.
        message_dict = response.choices[0].message.model_dump(exclude_unset=True)
        return LLMCompletionResponse(assistant_message=message_dict)
