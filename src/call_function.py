from src import FunctionDef, Prompt, build_function_calling_prompt, LLM


def _get_remaining_suffixes(
        functions_names: list[str],
        generated: str
        ) -> list[str]:
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
    return any(suffix.startswith(token_str) for suffix in remaining_suffixes)


def _mask_logits(
        masked_logits: list[float],
        logits: list[float],
        remaining_suffixes: list[str],
        llm: LLM
        ) -> None:
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
    full_prompt: str = build_function_calling_prompt(
        user_prompt, functions
    )

    functions_names: list[str] = [func.name for func in functions]
    input_ids: list[int] = llm.ft_encode(full_prompt)
    generated: str = ''

    while generated not in functions_names:

        remaining_suffixes: list[str] = _get_remaining_suffixes(
            functions_names, generated
            )

        logits: list[float] = llm.get_logits(input_ids)
        masked_logits: list[float] = [float('-inf')] * len(logits)
        _mask_logits(masked_logits, logits, remaining_suffixes, llm)

        best_logit: float = max(masked_logits)
        best_id: int = masked_logits.index(best_logit)
        best_token: str = llm.ft_decode(best_id)

        input_ids.append(best_id)
        generated += best_token

    return generated
