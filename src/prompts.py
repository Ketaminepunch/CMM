from src.models import FunctionDefinition


def build_function_list(definitions: list[FunctionDefinition]) -> str:

    lines = []
    for defn in definitions:
        param_parts = []
        for key, schema in defn.parameters.items():
            param_parts.append(f"{key}: {schema.type}")
        param_str = ",".join(param_parts)
        lines.append(f"{defn.name}({param_str}):{defn.description}")
    return "\n".join(lines)


def build_few_shot_examples() -> str:
    return (
        "Question: What is 12 plus 5?\n"
        'Answer: {"name": "demo_add", "parameters": {"x": 12, "y": 5}}\n\n'
        "Question: Say hello to Alex.\n"
        'Answer: {"name": "demo_greet", "parameters": {"person": "Alex"}}\n\n'
        "Question: Is 10 greater than 3?\n"
        'Answer: {"name": "demo_compare", "parameters": '
        '{"first": 10, "second": 3, "strict": true}}\n\n'
    )


def build_preamble(definitions: list[FunctionDefinition]) -> str:
    return (
        "Functions available:\n"
        + build_function_list(definitions)
        + "\n\n"
        + build_few_shot_examples()
    )


def build_prompt(preamble: str, question: str) -> str:
    return preamble + f"Question: {question}\nAnswer: "
