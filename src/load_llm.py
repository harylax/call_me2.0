from src import LLM
from pydantic import ValidationError  # type: ignore
import pyfiglet  # type: ignore
import time
import sys


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
        print(f"\033c\033[35m\n{s}\033[0m")
        time.sleep(0.5)


def _choose_llm() -> str:
    model: dict[int, str] = {
        1: 'Qwen/Qwen3-0.6B',
        2: 'HuggingFaceTB/SmolLM2-360M',
        3: 'Qwen/Qwen2.5-0.5B',
        4: 'Qwen/Qwen3.5-0.8B'}
    header: str = pyfiglet.figlet_format(
        'Call Me Maybe ...', font='dos_rebel')
    choice_menu: str = (
        "Choose model from:\n"
        "\t1 - Qwen/Qwen3-0.6B\n"
        "\t2 - HuggingFaceTB/SmolLM2-360M\n"
        "\t3 - Qwen/Qwen2.5-0.5B\n"
        "\t4 - Qwen/Qwen3.5-0.8B\n"
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
    _print_header()
    try:
        return LLM(model=_choose_llm())
    except ValidationError as err:
        for error in err.errors():
            print(f"ValidationError: {error['msg']}", file=sys.stderr)
        sys.exit(1)
