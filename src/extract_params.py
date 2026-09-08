from src import LLM, FunctionDef, build_params_prompt, Prompt


def _mask_logits(
        masked_logits: list[float],
        logits: list[float],
        param_type: str,
        llm: LLM,
        generated: str = '',
        seen: dict[int, int] | None = None
        ) -> None:
    if param_type == 'string':
        for token_id in llm.string_tokens:
            if seen is not None:
                if token_id in seen:
                    logits[token_id] -= 2.0 * seen[token_id]
            masked_logits[token_id] = logits[token_id]

    elif param_type == 'number':
        for token_id in llm.number_tokens:
            token_str = llm.ft_decode(token_id)
            if token_str[0] in ['-', '+']:
                if generated != '':
                    continue
            if '.' in token_str:
                if generated == '':
                    continue
                if '.' in generated:
                    continue
            masked_logits[token_id] = logits[token_id]

    elif param_type == 'integer':
        for token_id in llm.integer_tokens:
            token_str = llm.ft_decode(token_id)
            if token_str[0] in ['-', '+']:
                if generated != '':
                    continue
            masked_logits[token_id] = logits[token_id]


def _constrained_gen(
        param_type: str,
        llm: LLM,
        input_ids: list[int],
        closing_char: str,
        seen: dict[int, int] | None = None
        ) -> str:
    input_ids.extend(llm.ft_encode(closing_char))
    generated: str = ''
    while not generated.endswith(closing_char):
        logits: list[float] = llm.get_logits(input_ids)
        masked_logits: list[float] = [float('-inf')] * len(logits)
        _mask_logits(
            masked_logits, logits, param_type,
            llm, generated, seen
        )
        best_logit: float = max(masked_logits)
        best_id: int = masked_logits.index(best_logit)
        best_token: str = llm.ft_decode(best_id)

        input_ids.append(best_id)
        generated += best_token

        if seen is not None:
            if best_id in seen:
                seen[best_id] += 1
            else:
                seen[best_id] = 1

        if len(generated) > 50:
            break
    return generated.strip()


def params_from_llm(
        user_prompt: Prompt,
        llm: LLM,
        function: FunctionDef
        ) -> dict[str, str | int | float | bool]:
    full_prompt: str = build_params_prompt(
        user_prompt, function
    )
    input_ids: list[int] = llm.ft_encode(full_prompt)
    res: dict[str, str | int | float | bool] = {}
    for i, (param, param_def) in enumerate(function.parameters.items()):

        add_prompt: str = (
            f"\nThe parameter number {i} is "
            f"\"{param}\" and its type '{param_def.type}'\n"
            f"\n{param}="
            )
        add_token_ids: list[int] = llm.ft_encode(add_prompt)
        input_ids.extend(add_token_ids)

        if param_def.type == 'string':
            seen: dict[int, int] = {}
            generated: str = _constrained_gen(
                'string', llm, input_ids, '"', seen
            )
            res[param] = generated.rstrip('"').strip()

        elif param_def.type == 'number':
            generated = _constrained_gen(
                'number', llm, input_ids, "'"
            )
            res[param] = float(generated.rstrip("'"))

        elif param_def.type == 'integer':
            generated = _constrained_gen(
                'integer', llm, input_ids, "'"
            )
            res[param] = int(generated.rstrip("'"))

        elif param_def.type == 'boolean':
            logits = llm.get_logits(input_ids)
            res[param] = logits[llm.true_id] > logits[llm.false_id]

    return res
