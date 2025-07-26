#!/bin/bash

#
# A simple test runner script for the cli-assistant project.
#
# This script simplifies running the test suite by providing shortcuts for
# common testing scenarios.
#
# Usage:
#   ./run_tests.sh          - Run all tests.
#   ./run_tests.sh <name>   - Run a test module by name (e.g., 'agent' runs 'test_agent.py').
#   ./run_tests.sh <path>   - Run a specific test by its full path, e.g.,
#                             tests.test_agent.TestAgent.test_run_simple_completion

# Set the project root directory relative to the script's location.
# This allows the script to be run from anywhere.
PROJECT_ROOT=$(dirname "$0")/..

# Change to the project root to ensure all paths and imports work correctly.
cd "$PROJECT_ROOT" || exit

# Default command is to discover all tests in the 'tests' directory.
TEST_CMD="python3 -m unittest discover tests"

# If an argument is provided, try to find a matching test module or run a specific path.
if [ "$#" -gt 0 ]; then
    MODULE_NAME=$1
    TEST_FILE="test_${MODULE_NAME}.py"

    # Check if a file following the 'test_<name>.py' convention exists.
    if [ -f "tests/${TEST_FILE}" ]; then
        # If it exists, run discovery scoped to that file pattern.
        TEST_CMD="python3 -m unittest discover tests -p '${TEST_FILE}'"
    else
        # Otherwise, assume the argument is a full, specific test path.
        TEST_CMD="python3 -m unittest $1"
    fi
fi

echo "▶️  Executing: $TEST_CMD"
eval "$TEST_CMD"