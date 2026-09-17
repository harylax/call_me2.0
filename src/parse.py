"""Parse CLI args, function definitions and user prompts from JSON."""

from pydantic import BaseModel, ConfigDict, ValidationError
import json
from typing import Any, Literal
import sys
from argparse import ArgumentParser


class ParamDef(BaseModel):
    """Parameter type definition.

    Attributes:
        type: One of 'string', 'number', 'integer', or 'boolean'.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDef(BaseModel):
    """Function definition for function calling.

    Attributes:
        name: Function name.
        description: Function description.
        parameters: Mapping of parameter names to their definitions.
        returns: Return type definition.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParamDef]
    returns: ParamDef


class Prompt(BaseModel):
    """User prompt wrapper.

    Attributes:
        prompt: The user prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str


def _json_load(path: str) -> Any:
    """Load JSON data from a file.

    Args:
        path: Path to the JSON file.

    Returns:
        JSON content.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as err:
        print(
            "\033[31m"
            f"{err.__class__.__name__}: {err}"
            "\033[0m", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"\033[31mJSONDecodeError: {err}\033[0m", file=sys.stderr)
        sys.exit(1)


def parse_functions_definition(path: str) -> list[FunctionDef]:
    """Parse function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated FunctionDef objects.
    """
    res: list[FunctionDef] = []
    for func in _json_load(path):
        try:
            res.append(FunctionDef.model_validate(func))
        except ValidationError as err:
            for error in err.errors():
                print(
                    "\033[31m"
                    f"ValidationError: {error['msg']}"
                    "\033[0m", file=sys.stderr)
            sys.exit(1)
    if not res:
        print(
            "\033[31m"
            "Error: No functions definition available"
            "\033[0m", file=sys.stderr)
        sys.exit(1)
    return res


def parse_prompts(path: str) -> list[Prompt]:
    """Parse user prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        List of validated Prompt objects.
    """
    res: list[Prompt] = []
    for prompt in _json_load(path):
        try:
            res.append(Prompt.model_validate(prompt))
        except ValidationError as err:
            for error in err.errors():
                print(
                    "\033[31m"
                    f"ValidationError: {error['msg']}"
                    "\033[0m", file=sys.stderr)
            sys.exit(1)
    if not res:
        print(
            "\033[31m"
            "Error: No prompt provided"
            "\033[0m", file=sys.stderr)
    return res


def parse_args() -> tuple[str, str, str]:
    """Parse command-line arguments.

    Returns:
        Tuple of (input_path, functions_definition_path, output_path).
    """
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--input",
        default='data/input/function_calling_tests.json')
    parser.add_argument(
        "--functions_definition",
        default='data/input/functions_definition.json')
    parser.add_argument(
        "--output",
        default='data/output/function_calling_results.json')
    args = parser.parse_args()
    return args.input, args.functions_definition, args.output
