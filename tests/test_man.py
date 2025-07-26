import unittest
import subprocess
from unittest.mock import patch, MagicMock

from cli_assistant.ai.assistants import man


class TestManAssistant(unittest.TestCase):
    """Tests for the `man` assistant module."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.man._generate_man_summary")
    @patch("cli_assistant.ai.assistants.man.Console")
    @patch("cli_assistant.ai.assistants.man.Markdown")
    def test_man_displays_summary(self, MockMarkdown, MockConsole, mock_generate_summary):
        """Verify the main `man` function renders the generated summary using rich."""
        # Arrange
        summary_text = "# LS\n\nLists directory contents."
        mock_generate_summary.return_value = summary_text
        mock_console_instance = MockConsole.return_value
        mock_markdown_instance = MockMarkdown.return_value

        # Action
        man.man(self.mock_config, "ls")

        # Assert
        mock_generate_summary.assert_called_once_with(self.mock_config, "ls")
        MockMarkdown.assert_called_once_with(summary_text)
        mock_console_instance.print.assert_called_once_with(mock_markdown_instance)

    @patch("cli_assistant.ai.assistants.man.Agent")
    @patch("cli_assistant.ai.assistants.man._fetch_man_page")
    def test_generate_man_summary_success(self, mock_fetch_man_page, MockAgent):
        """Test summary generation when a man page is found."""
        # Arrange
        mock_fetch_man_page.return_value = "MAN PAGE CONTENT"
        mock_agent_instance = MockAgent.return_value
        mock_response = MagicMock()
        mock_response.content = "AI-generated summary"
        mock_agent_instance.run.return_value = mock_response

        # Action
        result = man._generate_man_summary(self.mock_config, "ls")

        # Assert
        mock_fetch_man_page.assert_called_once_with("ls")
        MockAgent.assert_called_once_with(self.mock_config, unittest.mock.ANY, man.SYSTEM_PROMPT)
        mock_agent_instance.run.assert_called_once()
        # Check that the man page content is in the prompt to the agent
        self.assertIn("MAN PAGE CONTENT", mock_agent_instance.run.call_args[0][0])
        self.assertEqual(result, "AI-generated summary")

    @patch("cli_assistant.ai.assistants.man.Agent")
    @patch("cli_assistant.ai.assistants.man._fetch_man_page")
    def test_generate_man_summary_no_man_page_found(self, mock_fetch_man_page, MockAgent):
        """Test behavior when `_fetch_man_page` returns nothing."""
        # Arrange
        mock_fetch_man_page.return_value = ""
        mock_agent_instance = MockAgent.return_value

        # Action
        result = man._generate_man_summary(self.mock_config, "nonexistent_command")

        # Assert
        mock_fetch_man_page.assert_called_once_with("nonexistent_command")
        mock_agent_instance.run.assert_not_called()
        self.assertEqual(result, "Could not find a man page for `nonexistent_command`.")

    @patch("subprocess.run")
    def test_fetch_man_page_success(self, mock_subprocess_run):
        """Test fetching a man page successfully."""
        # Arrange
        mock_result = MagicMock()
        mock_result.stdout = "This is the man page."
        mock_subprocess_run.return_value = mock_result

        # Action
        result = man._fetch_man_page("ls")

        # Assert
        mock_subprocess_run.assert_called_once_with(
            ["man", "ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        self.assertEqual(result, "This is the man page.")

    @patch("subprocess.run", side_effect=Exception("Command failed"))
    def test_fetch_man_page_failure(self, mock_subprocess_run):
        """Test fetching a man page when the subprocess call fails."""
        # Action
        result = man._fetch_man_page("ls")

        # Assert
        self.assertEqual(result, "")