"""Command-line entry point for the function-calling pipeline."""

from src import (
    timer, parse_args, run
)
import sys
import pyfiglet


@timer
def main() -> None:
    """Entry point: parse arguments and run the pipeline."""
    input_path, functions_definition_path, output_path = parse_args()
    try:
        run(input_path, functions_definition_path, output_path)
    except KeyboardInterrupt:
        print()
        print(pyfiglet.figlet_format('Program interrupted', font='slant'))
        sys.exit(1)


if __name__ == "__main__":
    main()
