"""Load and select the small LLM model used for function calling."""

from src import LLM
from pydantic import ValidationError
import pyfiglet
import time
import sys


def _print_header() -> None:
    """Print the animated startup header."""
    text: str = pyfiglet.figlet_format(
        'Call Me Maybe', font='dos_rebel')
    dot_1: str = pyfiglet.figlet_format(
        'Call Me Maybe .', font='dos_rebel')
    dot_2: str = pyfiglet.figlet_format(
        'Call Me Maybe ..', font='dos_rebel')
    dot_3: str = pyfiglet.figlet_format(
        'Call Me Maybe ...', font='dos_rebel')
    for s in [text, dot_1, dot_2, dot_3]:
        print(f"\033c\033[35m\n{s}\033[0m")
        time.sleep(0.2)


def _choose_llm() -> str:
    """Prompt the user to choose an LLM model.

    Returns:
        Selected model identifier string.
    """
    model: dict[int, str] = {
        1: 'Qwen/Qwen3-0.6B',
        2: 'HuggingFaceTB/SmolLM2-360M',
        3: 'Qwen/Qwen2.5-0.5B',
        4: 'Qwen/Qwen3-1.7B'}
    header: str = pyfiglet.figlet_format(
        'Call Me Maybe ...', font='dos_rebel')
    choice_menu: str = (
        "Choose model from:\n"
        "\t1 - Qwen/Qwen3-0.6B\n"
        "\t2 - HuggingFaceTB/SmolLM2-360M\n"
        "\t3 - Qwen/Qwen2.5-0.5B\n"
        "\t4 - Qwen/Qwen3-1.7B\n"
        "──────────────────────────────────────────────────────\n"
        "Your choice [1-4]: ")
    while True:
        try:
            print(f"\033c\033[35m\n{header}\033[0m")
            print(choice_menu, end='')
            choice: int = int(input())
            return model[choice]
        except (ValueError, KeyError):
            continue


def load_llm() -> LLM:
    """Prompt the user to choose an LLM model.

    Returns:
        Selected model identifier string.
    """
    _print_header()
    try:
        return LLM(model=_choose_llm())
    except ValidationError as err:
        for error in err.errors():
            print(
                "\033[31m"
                f"ValidationError: {error['msg']}"
                "\033[0m", file=sys.stderr)
        sys.exit(1)
