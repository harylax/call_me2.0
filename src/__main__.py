import json
from typing import Any
from src import (
    timer, parse_args, FunctionDef, parse_functions_definition, parse_prompts,
    function_name_from_llm, params_from_llm, get_function, LLM
)
from pathlib import Path
import sys
from pydantic import ValidationError  # type: ignore


@timer
def main() -> None:
    try:
        llm: LLM = LLM()
    except ValidationError as err:
        for error in err.errors():
            print(f"ValidationError: {error['msg']}", file=sys.stderr)
        sys.exit(1)

    input_path, functions_definition_path, output_path = parse_args()

    output: list[dict[str, Any]] = []

    functions: list[FunctionDef] = parse_functions_definition(
        functions_definition_path
        )

    for user_prompt in parse_prompts(input_path):
        llm_fn_name: str = function_name_from_llm(
            user_prompt, llm, functions
            )

        try:
            function: FunctionDef = get_function(llm_fn_name, functions)
        except ValueError as err:
            print(f"Value Error: {err}")
            raise SystemExit()

        llm_params: dict[str, str | int | float | bool] = params_from_llm(
            user_prompt, llm, function
        )

        output.append({
            'prompt': user_prompt.prompt,
            'name': llm_fn_name,
            'parameters': llm_params
        })
        print(f"prompt: {user_prompt.prompt}")
        print(f"name: {llm_fn_name}")
        print(f"parameters: {llm_params}")
    path: Path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    except OSError as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
