import unittest
from unittest.mock import patch
import sys

from io import StringIO
# This import will trigger the command registration via decorators in cli.py
from cli_assistant import cli


class TestCommandLineParser(unittest.TestCase):
    """Tests for the command-line argument parser in cli.py."""

    @patch("cli_assistant.cli._validate_ai_config")
    @patch("cli_assistant.cli.do")  # Patch 'do' where it is imported and used
    @patch("argcomplete.autocomplete")
    def test_do_command_parses_correctly(
        self, mock_autocomplete, mock_ai_do, mock_validate_config
    ):
        """Verify `assist do "prompt"` calls the correct handler with the right args."""
        # We pass only the arguments, not the program name, to the testable function.
        test_argv = ["do", "list all files"]
        cli.run_cli(test_argv)

        # Ensure config validation and the underlying AI function were called
        mock_validate_config.assert_called_once()
        mock_ai_do.assert_called_once()

        # Check that the AI function was called with the config and the correct prompt
        _config, prompt = mock_ai_do.call_args.args
        self.assertEqual(prompt, "list all files")

    @patch("cli_assistant.cli._validate_ai_config")
    @patch("cli_assistant.cli.explain")  # Patch 'explain' where it is used
    @patch("argcomplete.autocomplete")
    def test_explain_command_parses_correctly(
        self, mock_autocomplete, mock_ai_explain, mock_validate_config
    ):
        """Verify `assist explain "command"` calls the correct handler."""
        test_argv = ["explain", "ls -la"]
        cli.run_cli(test_argv)

        # Ensure config validation and the underlying AI function were called
        mock_validate_config.assert_called_once()
        mock_ai_explain.assert_called_once()

        # Check that the AI function was called with the config and the correct command
        _config, command_to_explain = mock_ai_explain.call_args.args
        self.assertEqual(command_to_explain, "ls -la")

    @patch("sys.stderr", new_callable=StringIO)
    @patch("argcomplete.autocomplete")
    def test_missing_required_argument_exits_with_error(
        self, mock_autocomplete, mock_stderr
    ):
        """Verify a command with a missing argument exits with a clear error."""
        test_argv = ["do"]  # Missing the 'prompt' argument

        with self.assertRaises(SystemExit) as cm:
            cli.run_cli(test_argv)

        self.assertIn("the following arguments are required: prompt", mock_stderr.getvalue())

    @patch("sys.stderr", new_callable=StringIO)
    @patch("argcomplete.autocomplete")
    def test_invalid_command_exits_with_error(
        self, mock_autocomplete, mock_stderr
    ):
        """Verify that an invalid command exits with a clear error."""
        test_argv = ["fly"]  # 'fly' is not a valid command

        with self.assertRaises(SystemExit) as cm:
            cli.run_cli(test_argv)

        self.assertIn("invalid choice: 'fly'", mock_stderr.getvalue())