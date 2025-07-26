import unittest
from unittest.mock import MagicMock, patch

from cli_assistant.ai.agent import Agent, Environment, LLMCompletionResponse
# Import the real LLMClient to access its static methods
from cli_assistant.ai.llm import LLMClient as RealLLMClient

class TestAgent(unittest.TestCase):
    """Tests for the Agent class."""

    def setUp(self):
        """Set up common test resources."""
        self.mock_config = {
            "provider": "mock_provider",
            "model": "mock_model",
            "provider_configs": {"mock_provider": {"api_key": "test-key"}},
        }
        self.system_prompt = "You are a test assistant."

        # When we patch `LLMClient`, the whole class is replaced by a mock,
        # including its static `format_*_message` methods. This causes tests to
        # fail because the agent's message list gets populated with mock objects
        # instead of the expected dictionaries.
        #
        # To fix this, we start the patcher manually in `setUp`, and then we
        # re-attach the *real* static methods to the mock class. This ensures
        # that the code under test behaves correctly while we still mock the
        # LLM's instance methods (like `completion`).
        patcher = patch("cli_assistant.ai.agent.LLMClient", autospec=True)
        self.MockLLMClient = patcher.start()
        self.addCleanup(patcher.stop)

        self.MockLLMClient.format_system_message = RealLLMClient.format_system_message
        self.MockLLMClient.format_user_message = RealLLMClient.format_user_message
        self.MockLLMClient.format_tool_message = RealLLMClient.format_tool_message

    def test_run_simple_completion(self):
        """Test a single run without tool calls, returning a final text response."""
        # Arrange: Mock the LLM to return a simple text response immediately.
        mock_llm_instance = self.MockLLMClient.return_value
        final_response_message = {"role": "assistant", "content": "Task complete."}
        mock_llm_instance.completion.return_value = LLMCompletionResponse(
            assistant_message=final_response_message
        )

        env = Environment()
        agent = Agent(self.mock_config, env, self.system_prompt)
        user_task = "Do a simple task."

        # Action
        response = agent.run(user_task, max_iterations=1)

        # Assert
        # 1. LLM completion was called once with the correct initial messages.
        mock_llm_instance.completion.assert_called_once()
        call_args = mock_llm_instance.completion.call_args
        self.assertEqual(call_args.kwargs["model"], "mock_provider:mock_model")

        expected_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_task},
        ]
        self.assertEqual(call_args.kwargs["messages"], expected_messages)

        # 2. The final response is the one returned by the LLM.
        self.assertEqual(response.content, "Task complete.")

        # 3. The agent's message history is correct.
        expected_history = expected_messages + [final_response_message]
        self.assertEqual(agent._messages, expected_history)

    def test_run_with_tool_call(self):
        """Test an agent run that involves a successful tool call."""
        # Arrange: Mock the LLM to first request a tool, then give a final answer.
        mock_llm_instance = self.MockLLMClient.return_value

        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"arg1": "value1"}'},
                }
            ],
        }
        final_response_message = {
            "role": "assistant",
            "content": "Tool executed, task complete.",
        }

        mock_llm_instance.completion.side_effect = [
            LLMCompletionResponse(assistant_message=tool_call_message),
            LLMCompletionResponse(assistant_message=final_response_message),
        ]

        # Mock the environment and its tool.
        env = Environment()
        env.run_tool = MagicMock(return_value="Tool result")
        env.get_tools = MagicMock(return_value=[{"type": "function"}])

        agent = Agent(self.mock_config, env, self.system_prompt)

        # Action
        response = agent.run("Use the test tool.", max_iterations=2)

        # Assert
        # 1. The environment's tool was called correctly.
        env.run_tool.assert_called_once_with("test_tool", {"arg1": "value1"})

        # 2. LLM completion was called twice.
        self.assertEqual(mock_llm_instance.completion.call_count, 2)

        # 3. The final response is the second one from the LLM.
        self.assertEqual(response.content, "Tool executed, task complete.")

    def test_run_with_tool_error(self):
        """Test an agent run where the tool execution fails."""
        # Arrange: Mock the LLM and an environment tool that raises an error.
        mock_llm_instance = self.MockLLMClient.return_value
        tool_call_message = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_456", "type": "function", "function": {"name": "failing_tool", "arguments": "{}"}}
            ],
        }
        mock_llm_instance.completion.side_effect = [
            LLMCompletionResponse(assistant_message=tool_call_message),
            LLMCompletionResponse(assistant_message={"role": "assistant", "content": "Fallback response."}),
        ]

        env = Environment()
        env.run_tool = MagicMock(side_effect=ValueError("Tool failed!"))
        env.get_tools = MagicMock(return_value=[{"type": "function"}])
        agent = Agent(self.mock_config, env, self.system_prompt)

        # Action
        agent.run("Run the failing tool.", max_iterations=2)

        # Assert: The tool message sent back to the LLM contains the error string.
        messages_for_second_call = mock_llm_instance.completion.call_args.kwargs["messages"]
        tool_result_message = messages_for_second_call[-1]

        self.assertEqual(tool_result_message["role"], "tool")
        self.assertEqual(tool_result_message["tool_call_id"], "call_456")
        self.assertIn("Error: Tool failed!", tool_result_message["content"])

    def test_run_stops_at_max_iterations(self):
        """Test that the agent stops when max_iterations is reached."""
        # Arrange: Mock an LLM that always requests a tool to create a loop.
        mock_llm_instance = self.MockLLMClient.return_value
        tool_call_message = {"role": "assistant", "tool_calls": [{"id": "call_loop", "type": "function", "function": {"name": "loop_tool", "arguments": "{}"}}]}
        mock_llm_instance.completion.return_value = LLMCompletionResponse(assistant_message=tool_call_message)

        env = Environment()
        env.run_tool = MagicMock(return_value="loop result")
        env.get_tools = MagicMock(return_value=[{"type": "function"}])
        agent = Agent(self.mock_config, env, self.system_prompt)

        # Action
        max_iters = 3
        agent.run("Loop forever", max_iterations=max_iters)

        # Assert: LLM completion was called exactly max_iterations times.
        self.assertEqual(mock_llm_instance.completion.call_count, max_iters)