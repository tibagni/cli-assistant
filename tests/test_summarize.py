import unittest
from unittest.mock import patch, MagicMock, mock_open

from cli_assistant.ai.assistants import summarize


class TestSummarizeAssistant(unittest.TestCase):
    """Tests for the main `summarize` assistant function."""

    def setUp(self):
        self.mock_config = {"provider": "test", "model": "test-model", "provider_configs": {}}

    @patch("cli_assistant.ai.assistants.summarize._read_sample_of_path")
    @patch("cli_assistant.ai.assistants.summarize.Agent")
    @patch("cli_assistant.ai.assistants.summarize.Console")
    def test_summarize_single_file(self, MockConsole, MockAgent, mock_read_sample):
        """Verify summarize handles a single file correctly."""
        # Arrange
        mock_read_sample.return_value = {"file": "test.py", "snippet": "print('hello')"}
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.content = "This is a summary."
        mock_console_instance = MockConsole.return_value

        # Action
        summarize.summarize(self.mock_config, ["test.py"])

        # Assert
        mock_read_sample.assert_called_once_with("test.py")
        mock_agent_instance.run.assert_called_once()

        # Check that the prompt sent to the agent is correct
        user_task_arg = mock_agent_instance.run.call_args[0][0]
        self.assertIn('"file": "test.py"', user_task_arg)
        self.assertIn('"snippet": "print(\'hello\')"', user_task_arg)

        mock_console_instance.print.assert_called_once()

    @patch("cli_assistant.ai.assistants.summarize._read_sample_of_path")
    @patch("cli_assistant.ai.assistants.summarize.Agent")
    def test_summarize_prompt_truncation(self, MockAgent, mock_read_sample):
        """Verify that the prompt is truncated if it exceeds the character limit."""
        # Arrange
        # Make each sample large enough that two will exceed the limit
        large_snippet = "a" * (summarize.PROMPT_CHAR_LIMIT // 2)
        mock_read_sample.side_effect = [
            {"file": "file1.txt", "snippet": large_snippet},
            {"file": "file2.txt", "snippet": large_snippet},
            {"file": "file3.txt", "snippet": "this should not be included"},
        ]
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.content = "Summary of truncated files."

        # Action
        with patch("cli_assistant.ai.assistants.summarize.Console"):  # Suppress output
            summarize.summarize(self.mock_config, ["file1.txt", "file2.txt", "file3.txt"])

        # Assert
        self.assertEqual(mock_read_sample.call_count, 2)  # Should stop after the second file
        mock_agent_instance.run.assert_called_once()

        user_task_arg = mock_agent_instance.run.call_args[0][0]
        self.assertIn("file1.txt", user_task_arg)
        self.assertNotIn("file3.txt", user_task_arg)
        self.assertIn("... (input truncated", user_task_arg)

    @patch("cli_assistant.ai.assistants.summarize.Agent")
    @patch("cli_assistant.ai.assistants.summarize.Console")
    def test_summarize_ai_failure(self, MockConsole, MockAgent):
        """Verify a fallback message is shown if the AI fails."""
        # Arrange
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value.content = None  # Simulate AI failure
        mock_console_instance = MockConsole.return_value

        # Action
        with patch(
            "cli_assistant.ai.assistants.summarize._read_sample_of_path",
            return_value={"file": "anyfile.txt", "snippet": "some content"},
        ):
            summarize.summarize(self.mock_config, ["anyfile.txt"])

        # Assert
        # Check that the fallback message is rendered
        mock_console_instance.print.assert_called_once()
        markdown_arg = mock_console_instance.print.call_args[0][0]
        self.assertIn("The AI failed to generate a summary.", markdown_arg.markup)


class TestSummarizeHelpers(unittest.TestCase):
    """Tests for the helper functions in the summarize assistant."""

    @patch("builtins.open", new_callable=mock_open, read_data="print('hello')")
    def test_read_sample_of_file_success(self, mock_file):
        """Test reading a snippet from a file successfully."""
        result = summarize._read_sample_of_file("test.py")
        self.assertEqual(result, {"file": "test.py", "snippet": "print('hello')"})
        mock_file.assert_called_once_with("test.py", "r", encoding="utf-8", errors="ignore")

    @patch("builtins.open", new_callable=mock_open, read_data="a" * 1000)
    def test_read_sample_of_file_truncation(self, mock_file):
        """Test that file snippets are truncated."""
        result = summarize._read_sample_of_file("large.txt", max_chars=800)
        self.assertTrue(result["snippet"].endswith("... (truncated)"))
        self.assertEqual(len(result["snippet"]), 800 + len("\n... (truncated)"))

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_read_sample_of_file_error(self, mock_file):
        """Test handling of file read errors."""
        result = summarize._read_sample_of_file("locked.txt")
        self.assertEqual(result, {"file": "locked.txt", "error": "Permission denied"})

    @patch("os.listdir", return_value=["file1.py", "sub_dir"])
    @patch("os.path.isfile", side_effect=[True, False])  # file1.py is a file, sub_dir is not
    @patch("cli_assistant.ai.assistants.summarize._read_sample_of_file")
    def test_read_sample_of_dir(self, mock_read_file, mock_isfile, mock_listdir):
        """Test reading a directory overview."""
        mock_read_file.return_value = {"file": "file1.py", "snippet": "..."}
        result = summarize._read_sample_of_dir("/fake/dir")
        mock_listdir.assert_called_once_with("/fake/dir")
        mock_read_file.assert_called_once_with("/fake/dir/file1.py")
        self.assertEqual(result["directory"], "/fake/dir")
        self.assertEqual(len(result["overview"]), 1)
        self.assertEqual(result["overview"][0]["file"], "file1.py")

    @patch("os.path.isfile", return_value=False)
    @patch("os.path.isdir", return_value=False)
    def test_read_sample_of_path_invalid(self, mock_isdir, mock_isfile):
        """Test _read_sample_of_path handles invalid paths."""
        result = summarize._read_sample_of_path("invalid/path")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "invalid/path doesn't appear to be valid")