"""Orchestrate the full function-calling pipeline and write results."""

from pathlib import Path
from typing import Any
from src import (
    Prompt, FunctionDef, function_name_from_llm, params_from_llm, get_function,
    load_llm, LLM, parse_functions_definition, parse_prompts)
import sys
import json


def build_output(
        output: list[dict[str, Any]],
        user_prompt: Prompt,
        llm: LLM,
        functions: list[FunctionDef]
        ) -> None:
    """Select function and parameters for one prompt and append to output.

    Args:
        output: List to append the result dict to.
        user_prompt: Current user prompt.
        llm: Initialized LLM instance.
        functions: Available function definitions.
    """
    try:
        llm_fn_name: str = function_name_from_llm(
            user_prompt, llm, functions)
        function: FunctionDef = get_function(llm_fn_name, functions)
        llm_params: dict[str, str | int | float | bool] = params_from_llm(
            user_prompt, llm, function)
        output.append({
            'prompt': user_prompt.prompt,
            'name': llm_fn_name,
            'parameters': llm_params})
    except Exception as err:
        print(
            "\033[31m"
            f"{err.__class__.__name__}: {err}"
            "\033[0m", file=sys.stderr)
        return
    print()
    print("══════════════════════════════════════════════════════")
    print(f"\033[34mprompt: \033[0m\t{user_prompt.prompt}")
    print(f"\033[32mname: \033[0m\t\t{llm_fn_name}")
    print(f"\033[35mparameters: \033[0m\t{llm_params}")
    print("══════════════════════════════════════════════════════")
    print()


def run(
    input_path: str,
    functions_definition_path: str,
    output_path: str
) -> None:
    """Run the full function-calling pipeline.

    Args:
        input_path: Path to the prompts JSON file.
        functions_definition_path: Path to the functions definition JSON.
        output_path: Path where results will be written.
    """
    llm: LLM = load_llm()

    output: list[dict[str, Any]] = []

    functions: list[FunctionDef] = parse_functions_definition(
        functions_definition_path)

    llm.cache_fn_name_tokens(functions)

    for user_prompt in parse_prompts(input_path):
        build_output(
            output, user_prompt, llm, functions
            )

    path: Path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    except OSError as err:
        print(
            "\033[31m"
            f"{err.__class__.__name__}: {err}"
            "\033[0m", file=sys.stderr)
