import unittest
from unittest.mock import patch, MagicMock, mock_open

from cli_assistant.ai.assistants import boilerplate
from cli_assistant.ai.assistants.boilerplate import BoilerplateEnvironment


class TestBoilerplateAssistant(unittest.TestCase):
    """Tests for the main `boilerplate` assistant function."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.boilerplate.BoilerplateEnvironment")
    @patch("cli_assistant.ai.assistants.boilerplate.Agent")
    @patch("cli_assistant.ai.assistants.boilerplate.Console")
    def test_boilerplate_orchestration(self, MockConsole, MockAgent, MockEnvironment):
        """Verify the main boilerplate function orchestrates agent and environment correctly."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.content = "Project created."
        mock_console_instance = MockConsole.return_value

        # Action
        boilerplate.boilerplate(self.mock_config, "a python project")

        # Assert
        MockEnvironment.assert_called_once()
        MockAgent.assert_called_once()
        mock_agent_instance.run.assert_called_once_with("a python project", max_iterations=10)

        # Check that status messages are printed
        self.assertIn(
            unittest.mock.call("[bold yellow]Starting boilerplate generation...[/]"),
            mock_console_instance.print.call_args_list,
        )
        self.assertIn(
            unittest.mock.call("\n[bold green]Boilerplate generation complete![/]"),
            mock_console_instance.print.call_args_list,
        )


class TestBoilerplateEnvironmentTools(unittest.TestCase):
    """Tests for the tools within the BoilerplateEnvironment."""

    def setUp(self):
        self.env = BoilerplateEnvironment()
        # Mock the console used by the tools to prevent output during tests
        self.env.console = MagicMock()

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_create_directory_success(self, mock_makedirs, mock_exists):
        """Test successful directory creation."""
        path = "new_project"
        result = self.env.create_directory(path)

        mock_exists.assert_called_once_with(path)
        mock_makedirs.assert_called_once_with(path)
        self.env.console.print.assert_called_once_with(f"[green]✓ Created directory:[/] {path}")
        self.assertEqual(result, f"Successfully created directory: {path}")

    @patch("os.path.exists", return_value=True)
    def test_create_directory_already_exists(self, mock_exists):
        """Test create_directory when path already exists."""
        path = "existing_project"
        result = self.env.create_directory(path)

        mock_exists.assert_called_once_with(path)
        self.assertEqual(result, f"Error: Path '{path}' already exists.")
        self.env.console.print.assert_not_called()

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs", side_effect=OSError("Permission denied"))
    def test_create_directory_os_error(self, mock_makedirs, mock_exists):
        """Test create_directory failure due to OS error."""
        path = "protected/new_project"
        result = self.env.create_directory(path)

        self.assertIn("Error creating directory", result)
        self.assertIn("Permission denied", result)
        self.env.console.print.assert_not_called()

    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_create_file_success(self, mock_file, mock_exists):
        """Test successful file creation."""
        path = "new_project/main.py"
        content = "print('hello')"
        result = self.env.create_file(path, content)

        mock_exists.assert_called_once_with(path)
        mock_file.assert_called_once_with(path, "w", encoding="utf-8")
        mock_file().write.assert_called_once_with(content)
        self.env.console.print.assert_called_once_with(f"[green]✓ Wrote file:[/] {path}")
        self.assertEqual(result, f"Successfully created file: {path}")

    @patch("os.path.exists", return_value=True)
    def test_create_file_already_exists(self, mock_exists):
        """Test create_file when path already exists."""
        path = "existing_project/main.py"
        result = self.env.create_file(path, "content")

        mock_exists.assert_called_once_with(path)
        self.assertEqual(result, f"Error: File '{path}' already exists.")
        self.env.console.print.assert_not_called()