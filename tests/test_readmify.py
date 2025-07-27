import unittest
from unittest.mock import patch, MagicMock, mock_open

from cli_assistant.ai.assistants import readmify
from cli_assistant.ai.assistants.readmify import ReadmifyEnvironment


class TestReadmifyAssistant(unittest.TestCase):
    """Tests for the main `readmify` assistant function."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.readmify.ReadmifyEnvironment")
    @patch("cli_assistant.ai.assistants.readmify.Agent")
    @patch("cli_assistant.ai.assistants.readmify.Console")
    def test_readmify_orchestration(self, MockConsole, MockAgent, MockEnvironment):
        """Verify the main readmify function orchestrates the agent and environment."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.interrupted = False  # Simulate a clean finish
        mock_console_instance = MockConsole.return_value

        # Action
        readmify.readmify(self.mock_config, "/fake/project")

        # Assert
        MockEnvironment.assert_called_once_with("/fake/project", self.mock_config)
        MockAgent.assert_called_once()
        mock_agent_instance.run.assert_called_once()
        mock_console_instance.print.assert_any_call(
            "[bold yellow]Analyzing project at '/fake/project' to generate README...[/]"
        )
        mock_console_instance.print.assert_any_call("\n[bold green]README generation process complete![/]")


class TestReadmifyEnvironment(unittest.TestCase):
    """Tests for the ReadmifyEnvironment and its tools."""

    def setUp(self):
        self.mock_config = {"provider": "test"}
        # We need to patch os functions before creating the environment instance
        # to control its behavior during __init__ and __del__.
        self.patcher_getcwd = patch("os.getcwd", return_value="/original/path")
        self.patcher_chdir = patch("os.chdir")
        self.mock_getcwd = self.patcher_getcwd.start()
        self.mock_chdir = self.patcher_chdir.start()
        self.addCleanup(self.patcher_getcwd.stop)
        self.addCleanup(self.patcher_chdir.stop)

    def test_environment_changes_directory_on_init_and_del(self):
        """Verify the environment changes CWD on creation and restores it on deletion."""
        # Action
        env = ReadmifyEnvironment("/fake/project", self.mock_config)

        # Assert init
        self.mock_chdir.assert_called_once_with("/fake/project")

        # Reset mock for del check
        self.mock_chdir.reset_mock()

        # Action del
        del env

        # Assert del
        self.mock_chdir.assert_called_once_with("/original/path")

    @patch("os.path.isdir", return_value=True)
    @patch("os.listdir", return_value=["file.txt", "subdir"])
    def test_list_files_and_dirs_success(self, mock_listdir, mock_isdir):
        """Test successful listing of files and directories."""
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()  # Mock console to prevent output
        result = env.list_files_and_dirs(".")

        mock_isdir.assert_called_once_with(".")
        mock_listdir.assert_called_once_with(".")
        self.assertEqual(result, "file.txt\nsubdir")
        env.console.print.assert_called_once()

    @patch("os.path.isfile", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="file content")
    def test_read_file_success(self, mock_open, mock_isfile):
        """Test successful file reading."""
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()
        result = env.read_file("file.txt")

        mock_isfile.assert_called_once_with("file.txt")
        mock_open.assert_called_once_with("file.txt", "r", encoding="utf-8", errors="ignore")
        self.assertEqual(result, "file content")
        env.console.print.assert_called_once()

    @patch("cli_assistant.ai.assistants.readmify._do_summarize")
    def test_summarize_path_tool(self, mock_do_summarize):
        """Test that summarize_path tool calls the underlying summarize function."""
        mock_do_summarize.return_value.content = "This is a summary."
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()
        result = env.summarize_path("some/path")

        mock_do_summarize.assert_called_once_with(self.mock_config, ["some/path"])
        self.assertEqual(result, "This is a summary.")
        env.console.print.assert_called_once()

    @patch("subprocess.run")
    def test_get_git_history_success(self, mock_run):
        """Test successful git history fetching."""
        mock_run.return_value.stdout = "commit1\ncommit2"
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()
        result = env.get_git_history(limit=5)

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0][2], "-n5")  # Check limit
        self.assertEqual(result, "commit1\ncommit2")
        env.console.print.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.input", return_value="y")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_readme_overwrite_confirmed(self, mock_open, mock_input, mock_exists):
        """Test writing README when overwrite is confirmed."""
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()
        result = env.write_readme("# New README")

        mock_exists.assert_called_once_with("README.md")
        mock_input.assert_called_once()
        mock_open.assert_called_once_with("README.md", "w", encoding="utf-8")
        mock_open().write.assert_called_once_with("# New README")
        self.assertIn("Successfully wrote README.md", result)
        env.console.print.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("builtins.input", return_value="n")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_readme_overwrite_denied(self, mock_open, mock_input, mock_exists):
        """Test writing README when overwrite is denied."""
        env = ReadmifyEnvironment("/fake/project", self.mock_config)
        env.console = MagicMock()
        result = env.write_readme("# New README")

        mock_exists.assert_called_once_with("README.md")
        mock_input.assert_called_once()
        mock_open.assert_not_called()
        self.assertEqual(result, "Aborted by user. README.md was not overwritten.")
        env.console.print.assert_not_called()