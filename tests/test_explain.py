import unittest
from unittest.mock import patch, MagicMock

from cli_assistant.ai.assistants import explain


class TestExplainAssistant(unittest.TestCase):
    """Tests for the `explain` assistant module."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.explain.Agent")
    @patch("cli_assistant.ai.assistants.explain.Console")
    @patch("cli_assistant.ai.assistants.explain.Markdown")
    def test_explain_success(self, MockMarkdown, MockConsole, MockAgent):
        """Verify that a successful AI response is rendered as markdown."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_response = MagicMock()
        mock_response.content = "This is the explanation."
        mock_agent_instance.run.return_value = mock_response

        mock_console_instance = MockConsole.return_value
        mock_markdown_instance = MockMarkdown.return_value

        # Action
        explain.explain(self.mock_config, "ls -la")

        # Assert
        # 1. Agent was created and run correctly
        MockAgent.assert_called_once_with(
            self.mock_config, unittest.mock.ANY, explain.SYSTEM_PROMPT
        )
        mock_agent_instance.run.assert_called_once_with(
            "Explain the following command: 'ls -la'", max_iterations=1
        )

        # 2. Rich was used to print the markdown
        MockMarkdown.assert_called_once_with("This is the explanation.")
        mock_console_instance.print.assert_called_once_with(mock_markdown_instance)

    @patch("cli_assistant.ai.assistants.explain.Agent")
    @patch("cli_assistant.ai.assistants.explain.Console")
    @patch("cli_assistant.ai.assistants.explain.Markdown")
    def test_explain_ai_failure(self, MockMarkdown, MockConsole, MockAgent):
        """Verify that an AI failure is handled and a fallback message is printed."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_response = MagicMock()
        mock_response.content = None  # Simulate AI failure
        mock_agent_instance.run.return_value = mock_response

        # Action
        explain.explain(self.mock_config, "some command")

        # Assert
        # Check that the fallback message is rendered
        MockMarkdown.assert_called_once_with("The AI failed to generate a description.")