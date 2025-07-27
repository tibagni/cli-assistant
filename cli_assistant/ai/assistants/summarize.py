import os
import json

from typing import Dict, List
from rich.console import Console
from rich.markdown import Markdown

from ..agent import Agent, Environment, LLMCompletionResponse

# A reasonable approximation for 1k tokens (1 token ~= 4 chars) to keep prompts concise.
PROMPT_CHAR_LIMIT = 4000
MAX_CHAR_LIMIT_PER_FILE = 800

SYSTEM_PROMPT = """
You are a helpful assistant that summarizes the content of files or directories.
If the input is a code file, provide a concise, high-level description of its purpose and main functions.
If the input is a markdown or text file, summarize the main points.
If the input is a directory, provide an overview of its purpose and list the main files and their roles
"""

def _read_sample_of_file(filepath: str, max_chars: int = MAX_CHAR_LIMIT_PER_FILE) -> Dict:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars)
            if len(content) == max_chars:
                content += "\n... (truncated)"
            return {"file": filepath, "snippet": content}
    except Exception as e:
        return {"file": filepath, "error": str(e)}

def _read_sample_of_dir(dirpath: str) -> Dict:
    file_summaries = []
    for fname in sorted(os.listdir(dirpath)):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            file_snippet = _read_sample_of_file(fpath)
            file_summaries.append(file_snippet)

    return {
        "directory": dirpath,
        "overview": file_summaries
    }

def _read_sample_of_path(path: str) -> Dict:
    if os.path.isfile(path):
        return _read_sample_of_file(path)
    elif os.path.isdir(path):
        return _read_sample_of_dir(path)
    
    return {"path": path, "error": f"{path} doesn't appear to be valid"}

def _do_summarize(config: Dict, paths: List[str]) -> LLMCompletionResponse:
    samples = []
    current_size = 0
    truncated = False

    for path in paths:
        sample = _read_sample_of_path(path)
        # Estimate the size of the sample as if it were in the final JSON.
        sample_size = len(json.dumps(sample))

        if current_size + sample_size > PROMPT_CHAR_LIMIT:
            truncated = True
            break  # Stop adding more files if we exceed the limit.

        samples.append(sample)
        current_size += sample_size

    prompt_content = json.dumps(samples, indent=2)
    if truncated:
        # Add a note to the AI that the input is incomplete.
        prompt_content += f'\n\n... (input truncated to fit within the prompt limit of {PROMPT_CHAR_LIMIT} characters)'

    user_task = f"Summarize the following files/directories:\n\n{prompt_content}"

    agent = Agent(config, Environment(), SYSTEM_PROMPT)
    return agent.run(user_task, max_iterations=1)

def summarize(config: Dict, paths: List[str]):
    """
    Reads content from multiple paths, sends it to the AI for summarization,
    and prints the result as markdown.
    """
    response = _do_summarize(config, paths)
    summary_markdown = response.content or "The AI failed to generate a summary."

    console = Console()
    markdown = Markdown(summary_markdown)
    console.print(markdown)