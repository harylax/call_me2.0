"""Utility helpers for vocabulary, function lookup and timing."""

from src import Small_LLM_Model, FunctionDef
import json
from collections.abc import Callable
import time
from functools import wraps
import sys


def get_vocab(llm: Small_LLM_Model) -> dict[str, int]:
    """Load and normalize the model vocabulary.

    Args:
        llm: The small LLM model instance.

    Returns:
        Dictionary mapping token strings to token IDs.
    """
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
    """Build inverted vocabulary (ID to token).

    Args:
        llm: The small LLM model instance.

    Returns:
        Dictionary mapping token IDs to token strings.
    """
    return {
        value: key
        for key, value
        in get_vocab(llm).items()}


def get_function(
        function_name: str,
        functions: list[FunctionDef]
        ) -> FunctionDef:
    """Find a function definition by name.

    Args:
        function_name: Name of the function to find.
        functions: List of available function definitions.

    Returns:
        The matching FunctionDef.

    Raises:
        ValueError: If the function name is not found.
    """
    for fn in functions:
        if fn.name == function_name:
            return fn
    raise ValueError(f"'{function_name}' not found in definitions")


def timer(func: Callable[[], None]) -> Callable[[], None]:
    """Measure and print execution time of a decorated function.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that prints elapsed time.
    """
    @wraps(func)
    def wrapper() -> None:
        start: float = time.perf_counter()
        func()
        end: float = time.perf_counter()
        minutes: int = int((end - start) // 60)
        seconds: int = int((end - start) % 60)
        print(f"Run completed in {minutes} minutes and {seconds} seconds")
    return wrapper
