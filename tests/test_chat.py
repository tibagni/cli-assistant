import unittest
from unittest.mock import patch, MagicMock

from cli_assistant.ai.assistants import chat
from cli_assistant.ai.assistants.chat import ChatEnvironment


class TestChatAssistant(unittest.TestCase):
    """Tests for the main `chat` assistant function."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.chat.ChatEnvironment")
    @patch("cli_assistant.ai.assistants.chat.Agent")
    @patch("builtins.input", return_value="hello agent")
    def test_chat_orchestration(self, mock_input, MockAgent, MockEnvironment):
        """Verify the main chat function orchestrates the agent and environment."""
        mock_agent_instance = MockAgent.return_value

        chat.chat(self.mock_config)

        MockEnvironment.assert_called_once_with(self.mock_config)
        MockAgent.assert_called_once()
        mock_agent_instance.run.assert_called_once_with("hello agent", max_iterations=10000)
        mock_input.assert_called_once_with("Ask anything: ")


class TestChatEnvironment(unittest.TestCase):
    """Tests for the ChatEnvironment and its tools."""

    def setUp(self):
        self.mock_config = {"provider": "test"}
        self.env = ChatEnvironment(self.mock_config)
        # Mock the console used by the tools to prevent output during tests
        self.env.console = MagicMock()

    @patch("os.getcwd", return_value="/fake/cwd")
    def test_get_current_working_directory(self, mock_getcwd):
        """Test successful retrieval of the current working directory."""
        result = self.env.get_current_working_directory()
        mock_getcwd.assert_called_once()
        self.assertEqual(result, "/fake/cwd")
        self.env.console.print.assert_called_once()

    @patch("os.path.isdir", side_effect=[True, False, True]) # First is for the argument, the other 2 are for the files
    @patch("os.path.isfile", side_effect=[True, False])
    @patch("os.listdir", return_value=["file.txt", "subdir"])
    def test_list_files_and_dirs_success(self, mock_listdir, mock_isfile, mock_isdir):
        """Test successful listing of files and directories."""
        result = self.env.list_files_and_dirs(".")
        self.assertEqual(result, "Files: file.txt\nFolders: subdir")
        self.env.console.print.assert_called_once()

    @patch("cli_assistant.ai.assistants.chat._do_summarize")
    def test_summarize_path_tool(self, mock_do_summarize):
        """Test that summarize_path tool calls the underlying summarize function."""
        mock_do_summarize.return_value.content = "This is a summary."
        result = self.env.summarize_path("some/path")

        mock_do_summarize.assert_called_once_with(self.mock_config, ["some/path"])
        self.assertEqual(result, "This is a summary.")
        self.env.console.print.assert_called_once()

    @patch("cli_assistant.ai.assistants.chat.ChatEnvironment._get_user_confirmation", return_value=True)
    @patch("subprocess.run")
    def test_run_command_success(self, mock_subprocess_run, mock_confirm):
        """Test successful command execution when user confirms."""
        mock_result = MagicMock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        result = self.env.run_command("ls -l")

        mock_confirm.assert_called_once_with("Run command 'ls -l'?")
        mock_subprocess_run.assert_called_once()
        self.assertEqual(result, "STDOUT:\ncommand output")
        self.env.console.print.assert_called_once()

    @patch("cli_assistant.ai.assistants.chat.ChatEnvironment._get_user_confirmation", return_value=False)
    @patch("subprocess.run")
    def test_run_command_denied(self, mock_subprocess_run, mock_confirm):
        """Test command is not run when user denies confirmation."""
        result = self.env.run_command("rm -rf /")
        mock_confirm.assert_called_once_with("Run command 'rm -rf /'?")
        mock_subprocess_run.assert_not_called()
        self.assertEqual(result, "Aborted by user. Command not executed.")
        self.env.console.print.assert_not_called()

    @patch("builtins.input", return_value="y")
    def test_get_user_confirmation_yes(self, mock_input):
        """Test user confirmation with 'y'."""
        result = self.env._get_user_confirmation("Confirm?")
        mock_input.assert_called_once_with("Confirm? [y/N] ")
        self.assertTrue(result)

    @patch("builtins.input", return_value="n")
    def test_get_user_confirmation_no(self, mock_input):
        """Test user confirmation with 'n'."""
        result = self.env._get_user_confirmation("Confirm?")
        self.assertFalse(result)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_handle_agent_response_interrupt(self, mock_input):
        """Test that handle_agent_response returns None on KeyboardInterrupt."""
        mock_response = MagicMock()
        mock_response.content = "Some AI response."
        result = self.env.handle_agent_response(mock_response)
        self.assertIsNone(result)

    @patch("builtins.input", return_value="a follow-up question")
    def test_handle_agent_response_with_input(self, mock_input):
        """Test that handle_agent_response returns the user's input."""
        mock_response = MagicMock()
        mock_response.content = "Some AI response."
        result = self.env.handle_agent_response(mock_response)
        self.assertEqual(result, "a follow-up question")

    def test_terminate_tool_with_message(self):
        """Test that the terminate tool sets the flag and prints the goodbye message."""

        self.assertFalse(self.env.should_terminate)

        self.env.terminate(goodbye_message="Session ended. Goodbye!")

        self.assertTrue(self.env.should_terminate)
        self.env.console.print.assert_called_once()

    def test_terminate_tool_without_message(self):
        """Test that the terminate tool sets the flag without printing a message."""

        self.assertFalse(self.env.should_terminate)
        self.env.terminate()

        self.assertTrue(self.env.should_terminate)
        self.env.console.print.assert_not_called()

    def test_handle_agent_response_stops_on_terminate_flag(self):
        """Verify handle_agent_response returns None if the terminate flag is set."""
        self.env.should_terminate = True
        result = self.env.handle_agent_response(MagicMock())
        self.assertIsNone(result)