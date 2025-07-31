import unittest
from unittest.mock import patch, mock_open

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


class TestConfigValidation(unittest.TestCase):
    """Tests for the _validate_ai_config function."""

    def setUp(self):
        """Reset the global config before each test."""
        cli._ai_config = {}

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"provider": "test"}')
    def test_config_exists_and_is_valid(self, mock_file, mock_exists):
        """Verify config is loaded correctly when the file exists and is valid."""
        cli._validate_ai_config()
        self.assertEqual(cli._ai_config, {"provider": "test"})
        # It should be idempotent and not re-read the file.
        mock_file.reset_mock()
        cli._validate_ai_config()
        mock_file.assert_not_called()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="{invalid json")
    @patch("sys.stderr", new_callable=StringIO)
    def test_config_exists_but_is_invalid(self, mock_stderr, mock_file, mock_exists):
        """Verify the application exits if the config file is malformed."""
        with self.assertRaises(SystemExit) as cm:
            cli._validate_ai_config()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Error reading or parsing", mock_stderr.getvalue())

    @patch("os.path.exists", return_value=False)
    @patch("builtins.input", return_value="n")
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.stderr", new_callable=StringIO)
    def test_config_creation_denied_by_user(self, mock_stderr, mock_stdout, mock_input, mock_exists):
        """Verify the application exits if the user denies config creation."""
        with self.assertRaises(SystemExit) as cm:
            cli._validate_ai_config()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Configuration file not found", mock_stdout.getvalue())
        self.assertIn("Configuration is required", mock_stderr.getvalue())

    @patch("os.path.exists", return_value=False)
    @patch("builtins.input", return_value="y")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("os.getenv", return_value="my-editor")
    # The subprocess.run call is already mocked, which prevents the editor from opening.
    # We add stdout patching to silence the print statements during the test run.
    @patch("subprocess.run") 
    @patch("sys.stdout", new_callable=StringIO)
    @patch("json.load")
    def test_config_creation_flow_success(
        self, mock_json_load, mock_stdout, mock_subprocess, mock_getenv, mock_json_dump, mock_file, mock_makedirs, mock_input, mock_exists
    ):
        """Verify the full config creation flow when the user confirms."""
        # Arrange: Assume the user saves a valid config, so json.load will succeed.
        mock_json_load.return_value = {"provider": "test"}

        # Action
        cli._validate_ai_config()

        # Assertions for the creation flow
        mock_exists.assert_called_once()
        mock_input.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_subprocess.assert_called_once()

        # Assert that the template was written
        mock_file.assert_any_call(unittest.mock.ANY, "w")
        mock_json_dump.assert_called_once()
        config_template = mock_json_dump.call_args[0][0]
        self.assertEqual(config_template["provider"], "openai")

        # Assert that the config was loaded at the end of the same function call
        mock_json_load.assert_called_once()
        self.assertEqual(cli._ai_config, {"provider": "test"})

    @patch("os.path.exists", return_value=False)
    @patch("builtins.input", return_value="y")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.getenv", return_value="bad-editor")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.stderr", new_callable=StringIO)
    def test_config_creation_editor_not_found(self, mock_stderr, mock_stdout, mock_subprocess, mock_getenv, mock_file, mock_makedirs, mock_input, mock_exists):
        """Verify the application exits if the specified editor is not found."""
        with self.assertRaises(SystemExit) as cm:
            cli._validate_ai_config()
        self.assertEqual(cm.exception.code, 1)
        mock_subprocess.assert_called_once()
        self.assertIn("Could not find editor", mock_stderr.getvalue())