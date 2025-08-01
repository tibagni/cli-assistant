[![Python application](https://github.com/tibagni/cli-assistant/actions/workflows/python-app.yml/badge.svg)](https://github.com/tibagni/cli-assistant/actions/workflows/python-app.yml)

# CLI Assistant

This is a personal project created to better understand how to integrate Large Language Models (LLMs) into a command-line application. It's an AI-powered assistant designed to help with various development tasks.

## Features

`assist` is an AI-driven command-line tool that can help streamline your workflow. It can generate shell commands, explain complex syntax, summarize code, create project boilerplate, and even generate README files for you.

- **`assist do <prompt>`**: Generates and executes shell commands from natural language.
- **`assist explain <command>`**: Explains a shell command in detail.
- **`assist man <command>`**: Provides a summarized, easy-to-read man page.
- **`assist summarize <paths...>`**: Summarizes files and directories.
- **`assist boilerplate <description>`**: Generates project scaffolding from a description.
- **`assist readmify [path]`**: Generates a README.md file for a project directory.
- **`assist chat`**: Starts an interactive chat session with a tool-equipped AI.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tibagni/cli-assistant.git
    cd assistant-cli
    ```

2.  **Install in editable mode:**
    This will install the `assist` command and all its dependencies.
    ```bash
    pip install -e .
    ```

## Configuration

The first time you run any `assist` command, the application will check for a configuration file at `~/.cli-assist/config.json`.

If the file is not found, it will prompt you to create one. Upon confirmation, it will open a template file in your default editor (`$EDITOR`). You just need to add your API key.

The default configuration looks like this:

    ```json
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "provider_configs": {"openai": {"api_key": "YOUR_API_KEY_HERE"}}
    }
    ```
    Replace `"YOUR_API_KEY_HERE"` with your actual API key and save the file to complete the setup.

## Shell Auto-Completion (Recommended)

To enable tab-completion for commands and arguments, you need to register `argcomplete`.

1.  **Run the registration command:**
    ```bash
    eval "$(register-python-argcomplete assist)"
    ```

2.  **Make it permanent:**
    To enable completion for all future terminal sessions, add the command above to your shell's startup file (e.g., `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`).

## Running Tests

This project uses Python's built-in `unittest` framework. A helper script is provided to simplify running tests.
- **Run all tests:**
  ```bash
  ./tests/run_tests.sh
  ```

- **Run a specific test suite (e.g., `chat` tests):**
  ```bash
  ./tests/run_tests.sh chat
  ```

- **Run a single test case:**
  ```bash
  ./tests/run_tests.sh tests.test_agent.TestAgent.test_run_simple_completion
  ```
