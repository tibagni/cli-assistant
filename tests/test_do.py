import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

from cli_assistant.ai.assistants import do
from cli_assistant.ai.assistants.do import CommandSuggestion


class TestDoAssistant(unittest.TestCase):
    """Tests for the `do` assistant module."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.do.Agent")
    def test_suggest_shell_command_success(self, MockAgent):
        """Verify that a valid JSON response is parsed into a CommandSuggestion object."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_response = MagicMock()
        mock_response.content = '{"command": "ls -l", "risk_assessment": 0, "explanation": "list files", "disclaimer": ""}'
        mock_agent_instance.run.return_value = mock_response

        # Action
        result = do._suggest_shell_command(self.mock_config, "list all files")

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CommandSuggestion)
        self.assertEqual(result.command, "ls -l")
        self.assertEqual(result.risk_assessment, 0)
        mock_agent_instance.run.assert_called_once_with(
            "list all files",
            max_iterations=1,
            response_format={"type": "json_schema", "json_schema": do.COMMAND_SUGGESTION_SCHEMA},
        )

    @patch("cli_assistant.ai.assistants.do.Agent")
    def test_suggest_shell_command_invalid_json(self, MockAgent):
        """Verify that malformed JSON from the AI is handled gracefully."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_response = MagicMock()
        mock_response.content = '{"command": "ls -l", "risk_assessment": 0, '  # Malformed
        mock_agent_instance.run.return_value = mock_response

        # Action
        result = do._suggest_shell_command(self.mock_config, "list all files")

        # Assert
        self.assertEqual(result.explanation, "Error: The AI failed to return a valid command.")
        self.assertEqual(result.command, "")

    @patch("cli_assistant.ai.assistants.do._suggest_shell_command")
    @patch("builtins.input", return_value="y")
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_do_user_confirms_execution(self, mock_stdout, mock_os_system, mock_input, mock_suggest):
        """Test the full flow where the user confirms and the command is run."""
        # Arrange
        suggestion = CommandSuggestion(
            command="echo 'hello'", risk_assessment=0, explanation="prints hello", disclaimer=""
        )
        mock_suggest.return_value = suggestion

        # Action
        do.do(self.mock_config, "say hello")

        # Assert
        mock_suggest.assert_called_once_with(self.mock_config, "say hello")
        self.assertIn("Suggested command:\n  echo 'hello'", mock_stdout.getvalue())
        self.assertIn("Explanation:\n  prints hello", mock_stdout.getvalue())
        mock_os_system.assert_called_once_with("echo 'hello'")

    @patch("cli_assistant.ai.assistants.do._suggest_shell_command")
    @patch("builtins.input", return_value="n")
    @patch("os.system")
    @patch("sys.stdout", new_callable=StringIO)
    def test_do_user_denies_execution(self, mock_stdout, mock_os_system, mock_input, mock_suggest):
        """Verify that the command is not run if the user denies confirmation."""
        # Arrange
        suggestion = CommandSuggestion(
            command="rm -rf /", risk_assessment=2, explanation="deletes everything", disclaimer="Danger!"
        )
        mock_suggest.return_value = suggestion

        # Action
        do.do(self.mock_config, "delete all files")

        # Assert
        mock_os_system.assert_not_called()

    @patch("cli_assistant.ai.assistants.do._suggest_shell_command")
    @patch("sys.stdout", new_callable=StringIO)
    def test_do_shows_disclaimer_for_risky_command(self, mock_stdout, mock_suggest):
        """Verify that the disclaimer is shown for commands with risk > 0."""
        # Arrange
        suggestion = CommandSuggestion(
            command="sudo rm -rf /", risk_assessment=2, explanation="deletes everything", disclaimer="This is very dangerous"
        )
        mock_suggest.return_value = suggestion

        # Mock input to prevent the test from hanging, and run the function.
        with patch("builtins.input", return_value="n"):
            do.do(self.mock_config, "delete all files")

        # Assert
        self.assertIn("⚠️  Disclaimer:\n  This is very dangerous", mock_stdout.getvalue())