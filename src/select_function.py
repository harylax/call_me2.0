"""Select the function name via constrained LLM decoding."""

from src import FunctionDef, Prompt, build_function_calling_prompt, LLM


def _get_remaining_suffixes(
        functions_names: list[str],
        generated: str
        ) -> list[str]:
    """Get remaining suffixes of function names that match the prefix.

    Args:
        functions_names: List of function names.
        generated: Currently generated prefix.

    Returns:
        List of remaining suffixes.
    """
    remaining_suffixes: list[str] = []
    for name in functions_names:
        if name.startswith(generated):
            step: int = len(generated)
            suffix: str = name[step:]
            remaining_suffixes.append(suffix)
    return remaining_suffixes


def _is_token_prefix_of_any_suffix(
        token_str: str,
        remaining_suffixes: list[str]
        ) -> bool:
    """Check if a token is a prefix of any remaining suffix.

    Args:
        token_str: Token string to check.
        remaining_suffixes: List of remaining suffixes.

    Returns:
        True if the token is a valid prefix of any suffix.
    """
    return any(suffix.startswith(token_str) for suffix in remaining_suffixes)


def _mask_logits(
        masked_logits: list[float],
        logits: list[float],
        remaining_suffixes: list[str],
        llm: LLM
        ) -> None:
    """Mask logits to only allow tokens that continue valid function names.

    Args:
        masked_logits: Output list to fill with masked values.
        logits: Original logits from the model.
        remaining_suffixes: Valid remaining suffixes.
        llm: LLM instance providing token sets.
    """
    for token_id in llm.fn_name_tokens:
        token_str: str = llm.ft_decode(token_id)
        if _is_token_prefix_of_any_suffix(
            token_str, remaining_suffixes
        ):
            masked_logits[token_id] = logits[token_id]


def function_name_from_llm(
        user_prompt: Prompt,
        llm: LLM,
        functions: list[FunctionDef]
        ) -> str:
    """Generate the function name via constrained decoding.

    Args:
        user_prompt: User prompt.
        llm: Initialized LLM instance.
        functions: Available function definitions.

    Returns:
        Selected function name as a string.
    """
    full_prompt: str = build_function_calling_prompt(
        user_prompt, functions)

    functions_names: list[str] = [func.name for func in functions]
    input_ids: list[int] = llm.ft_encode(full_prompt)
    generated: str = ''

    ##########################################
    cache, logits = llm.get_logits(input_ids)
    ##########################################

    while generated not in functions_names:

        remaining_suffixes: list[str] = _get_remaining_suffixes(
            functions_names, generated)
        # logits: list[float] = llm.get_logits(input_ids)
        masked_logits: list[float] = [float('-inf')] * len(logits)
        _mask_logits(masked_logits, logits, remaining_suffixes, llm)

        best_logit: float = max(masked_logits)
        best_id: int = masked_logits.index(best_logit)
        best_token: str = llm.ft_decode(best_id)

        input_ids.append(best_id)
        generated += best_token

        ######################################
        cache, logits = llm.get_logits([best_id], cache)
        ######################################

        print(
            f"\r\033[36mGenerating function name...\033[0m "
            f"\033[32m{generated}\033[0m",
            end='', flush=True)

    return generated
