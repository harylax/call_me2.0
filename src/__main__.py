import json
from typing import Any
from src import (
    timer, parse_args, FunctionDef, parse_functions_definition, parse_prompts,
    function_name_from_llm, params_from_llm, get_function, LLM,
    Prompt
)
from pathlib import Path
import sys
from pydantic import ValidationError  # type: ignore
import pyfiglet  # type: ignore
import time
import threading


def _print_header() -> None:
    text: str = pyfiglet.figlet_format(
        'Call Me Maybe', font='dos_rebel')
    dot_1: str = pyfiglet.figlet_format(
        'Call Me Maybe .', font='dos_rebel')
    dot_2: str = pyfiglet.figlet_format(
        'Call Me Maybe ..', font='dos_rebel')
    dot_3: str = pyfiglet.figlet_format(
        'Call Me Maybe ...', font='dos_rebel')
    for s in [text, dot_1, dot_2, dot_3]:
        print(
            "\033c"
            "\033[35m\n"
            f"{s}"
            "\033[0m")
        time.sleep(0.5)


def load_llm() -> LLM:
    _print_header()
    try:
        return LLM()
    except ValidationError as err:
        for error in err.errors():
            print(f"ValidationError: {error['msg']}", file=sys.stderr)
        sys.exit(1)


def build_output(
        output: list[dict[str, Any]],
        user_prompt: Prompt,
        llm: LLM,
        functions: list[FunctionDef]
        ) -> None:
    is_ready: bool = False
    bar_done: Any = threading.Event()

    def compute() -> None:
        nonlocal is_ready

        llm_fn_name: str = function_name_from_llm(user_prompt, llm, functions)
        try:
            function: FunctionDef = get_function(llm_fn_name, functions)
        except ValueError as err:
            print(f"ValueError: {err}", file=sys.stderr)
            sys.exit(1)

        llm_params: dict[str, str | int | float | bool] = params_from_llm(
            user_prompt, llm, function)
        output.append({
            'prompt': user_prompt.prompt,
            'name': llm_fn_name,
            'parameters': llm_params})
        is_ready = True
        bar_done.wait()
        print()
        print(f"\033[34mprompt: \033[0m\t{user_prompt.prompt}")
        print(f"\033[32mname: \033[0m\t\t{llm_fn_name}")
        print(f"\033[35mparameters: \033[0m\t{llm_params}")
        print()

    def progress_bar() -> None:
        nonlocal is_ready
        bar_len: int = 40
        i: int = 0
        while not is_ready:
            bar: str = "█" * i + "░" * (bar_len - i)
            print(
                f"\r\033[36mGenerating|\033[0m{bar}\033[36m|\033[0m",
                end="", flush=True)
            time.sleep(0.15)
            i = (i + 1) % bar_len
        bar = "█" * bar_len
        print(
            f"\r\033[36mGenerating|\033[0m{bar}\033[36m|\033[0m", flush=True)
        time.sleep(0.2)
        bar_done.set()

    thread_compute: Any = threading.Thread(
        target=compute, daemon=True)
    thread_bar: Any = threading.Thread(
        target=progress_bar, daemon=True)

    thread_compute.start()
    thread_bar.start()

    thread_compute.join()
    thread_bar.join()


def run(
    input_path: str,
    functions_definition_path: str,
    output_path: str
) -> None:
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
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)


@timer
def main() -> None:
    input_path, functions_definition_path, output_path = parse_args()
    try:
        run(input_path, functions_definition_path, output_path)
    except KeyboardInterrupt:
        print()
        print(pyfiglet.figlet_format('Program interrupted', font='slant'))
        sys.exit(1)


if __name__ == "__main__":
    main()
