from src import Small_LLM_Model, FunctionDef
import json
from collections.abc import Callable
import time
from functools import wraps
import sys


def get_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    vocab: dict[str, int] = {}
    try:
        with open(llm.get_path_to_vocab_file()) as f:
            vocab = json.load(f)
    except OSError as err:
        print(
            f"{err.__class__.__name__}: {err}",
            file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(
            f"JSONDecodeError: {err}",
            file=sys.stderr)
        sys.exit(1)
    return {
        key.replace(
            'Ġ', ' '
            ).replace(
                'Ċ', '\n'
                ).replace(
                    'ĉ', '\t'
                    ).replace(
                        'Ď', '\r'
                        ): value
        for key, value
        in vocab.items()}


def get_inverted_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    return {
        value: key
        for key, value
        in get_vocab(llm).items()}


def get_function(
        function_name: str,
        functions: list[FunctionDef]
        ) -> FunctionDef:
    for fn in functions:
        if fn.name == function_name:
            return fn
    raise ValueError(f"'{function_name}' not found in definitions")


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper() -> None:
        start: float = time.perf_counter()
        func()
        end: float = time.perf_counter()
        minutes: int = int((end - start) // 60)
        seconds: int = int((end - start) % 60)
        print(f"Run completed in {minutes} minutes and {seconds} seconds")
    return wrapper
