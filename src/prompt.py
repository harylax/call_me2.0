from src import FunctionDef, Prompt


def _functions_definition(functions: list[FunctionDef]) -> str:
    res: list[str] = []
    for func in functions:
        params_str: str = ', '.join(
            f"{key}: {value}"
            for key, value
            in func.parameters.items()
            )
        res.append(
            f" - function name: {func.name}, "
            f"parameter(s): {params_str}, "
            f"returns: {func.returns}, "
            f"description: {func.description}"
            )
    return '\n'.join(fn for fn in res)


def build_function_calling_prompt(
        prompt: Prompt,
        functions: list[FunctionDef]
        ) -> str:
    return (
        "You are a function calling assistant...\n\n"
        f"Available functions:\n{_functions_definition(functions)}\n\n"
        f"User main prompt: {prompt.prompt}\n"
        "Function to call: "
    )


def build_params_prompt(
        prompt: Prompt,
        function: FunctionDef
        ) -> str:
    params_str: str = ', '.join(
        f"parameter '{key}' (type: {value})"
        for key, value
        in function.parameters.items()
        )
    return (
        "You are a parameters extractor assistant...\n\n"
        f"Function called: {_functions_definition([function])}\n"
        f"User main prompt: {prompt.prompt}\n"
        f"Extract the value of: {params_str}.\n"
        "Don't add extra informations.\n"
        "Avoid repetition of tokens unless necessary.\n"
        "Preserve spelling, capitalisation, uppercase, lowercase, "
        "mixed case, numbers in the user main prompt.\n"
        "As reminder, vowels are: 'aeiouAEIOU'.\n"
        "Parameter(s) value(s) extraction:\n"
    )