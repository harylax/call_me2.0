from pydantic import BaseModel, ConfigDict, ValidationError  # type: ignore
import json
from typing import Any, Literal
import sys
from argparse import ArgumentParser


class ParamDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParamDef]
    returns: ParamDef


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str


def _json_load(path: str) -> list[dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as err:
        print(f"{err.__class__.__name__}: {err}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err}")
        sys.exit(1)


def parse_functions_definition(path: str) -> list[FunctionDef]:
    res: list[FunctionDef] = []
    for func in _json_load(path):
        try:
            res.append(FunctionDef.model_validate(func))
        except ValidationError as err:
            for error in err.errors():
                print(f"ValidationError: {error['msg']}", file=sys.stderr)
            sys.exit(1)
    return res


def parse_prompts(path: str) -> list[Prompt]:
    res: list[Prompt] = []
    for prompt in _json_load(path):
        try:
            res.append(Prompt.model_validate(prompt))
        except ValidationError as err:
            for error in err.errors():
                print(f"ValidationError: {error['msg']}", file=sys.stderr)
            sys.exit(1)
    return res


def parse_args() -> tuple[str, str, str]:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--input",
        default='data/input/function_calling_tests.json'
        )
    parser.add_argument(
        "--functions_definition",
        default='data/input/functions_definition.json'
        )
    parser.add_argument(
        "--output",
        default='data/output/function_calling_results.json'
        )
    args = parser.parse_args()
    return args.input, args.functions_definition, args.output
