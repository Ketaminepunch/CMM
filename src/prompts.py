"""Builds the text prompt fed to the model before constrained decoding.

The preamble (function list + few-shot examples) only depends on the
function catalog, so it's built once per run; each individual prompt
is then appended to it fresh.
"""

from src.models import FunctionDefinition


def build_function_list(definitions: list[FunctionDefinition]) -> str:
    """Render each function as a one-line ``name(params):description``."""
    lines = []
    for defn in definitions:
        param_parts = []
        for key, schema in defn.parameters.items():
            param_parts.append(f"{key}: {schema.type}")
        param_str = ",".join(param_parts)
        lines.append(f"{defn.name}({param_str}):{defn.description}")
    return "\n".join(lines)


def build_few_shot_examples() -> str:
    """Return a fixed block of example Question/Answer pairs.

    These teach the model the expected JSON call format; the
    function names used here are illustrative only and don't need
    to exist in the real catalog.
    """
    return (
        "Question: What is 12 plus 5?\n"
        'Answer: {"name": "demo_add", "parameters": {"x": 12, "y": 5}}\n\n'
        "Question: Is 10 greater than 3?\n"
        'Answer: {"name": "demo_compare", "parameters": '
        '{"first": 10, "second": 3, "strict": true}}\n\n'
        "Question: Run the query 'DELETE FROM tmp' on the staging database\n"
        'Answer: {"name": "demo_run_query", "parameters": '
        '{"query": "DELETE FROM tmp", "database": "staging"}}\n\n'
        "Question: Read the file at /var/log/app.log with ascii encoding\n"
        'Answer: {"name": "demo_read_file", "parameters": '
        '{"path": "/var/log/app.log", "encoding": "ascii"}}\n\n'
        "Question: Read D:\\Backup\\notes.txt with ascii encoding\n"
        'Answer: {"name": "demo_read_file", "parameters": '
        '{"path": "D:\\\\Backup\\\\notes.txt", "encoding": "ascii"}}\n\n'
        'Question: Format template: Reply "yes" to {sender} now!\n'
        'Answer: {"name": "demo_format", "parameters": '
        '{"template": "Reply \\"yes\\" to {sender} now!"}}\n\n'
    )


def build_preamble(definitions: list[FunctionDefinition]) -> str:
    """Build the shared preamble: function list plus few-shot examples."""
    return (
        "Functions available:\n"
        + build_function_list(definitions)
        + "\n\n"
        + build_few_shot_examples()
    )


def build_prompt(preamble: str, question: str) -> str:
    """Append one question to the shared preamble to form a full prompt."""
    return preamble + f"Question: {question}\nAnswer: "
