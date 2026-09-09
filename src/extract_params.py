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
        if generated == '':
            for token_id in llm.start_number_tokens:
                masked_logits[token_id] = logits[token_id]
        else:
            for token_id in llm.mid_number_tokens:
                masked_logits[token_id] = logits[token_id]

    elif param_type == 'integer':
        if generated == '':
            for token_id in llm.start_integer_tokens:
                masked_logits[token_id] = logits[token_id]
        else:
            for token_id in llm.mid_integer_tokens:
                masked_logits[token_id] = logits[token_id]


def _list_signed_digits_in_prompt(prompt: str) -> list[str]:
    res: list[str] = []
    for i in range(1, len(prompt)):
        if prompt[i].isdigit() and prompt[i - 1] in ('+', '-'):
            s: str = prompt[i - 1] + prompt[i]
            res.append(s)
    return res


def _constrained_gen(
        param_type: str,
        llm: LLM,
        input_ids: list[int],
        closing_char: str,
        seen: dict[int, int] | None = None,
        signed_list: list[str] = []
        ) -> str:
    input_ids.extend(llm.ft_encode(closing_char))
    generated: str = ''
    #########################################
    cache, logits = llm.get_logits(input_ids)
    #########################################
    while not generated.endswith(closing_char):
        masked_logits: list[float] = [float('-inf')] * len(logits)
        _mask_logits(
            masked_logits, logits, param_type,
            llm, generated, seen
        )
        # logits: list[float] = llm.get_logits(input_ids)
        best_logit: float = max(masked_logits)
        best_id: int = masked_logits.index(best_logit)
        best_token: str = llm.ft_decode(best_id)

        input_ids.append(best_id)
        if (
            param_type in ['number', 'integer']
            and generated == ''
            and signed_list
        ):
            if signed_list[0].startswith('+'):
                if best_token[0] == signed_list[0][1]:
                    signed_list.pop(0)
            elif signed_list[0].startswith('-'):
                if best_token[0] == signed_list[0][1]:
                    best_token = '-' + best_token
                    signed_list.pop(0)
        generated += best_token

        if seen is not None:
            if best_id in seen:
                seen[best_id] += 1
            else:
                seen[best_id] = 1

        if len(generated) > 50:
            break

        ################################################
        cache, logits = llm.get_logits([best_id], cache)
        ################################################
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
    signed_list: list[str] = _list_signed_digits_in_prompt(
            user_prompt.prompt
            )
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
                'number', llm, input_ids, "'",
                signed_list=signed_list
            )
            res[param] = float(generated.rstrip("'"))

        elif param_def.type == 'integer':
            generated = _constrained_gen(
                'integer', llm, input_ids, "'",
                signed_list=signed_list
            )
            res[param] = int(generated.rstrip("'"))

        elif param_def.type == 'boolean':
            # logits: list[float] = llm.get_logits(input_ids)
            #####################################
            _, logits = llm.get_logits(input_ids)
            #####################################
            res[param] = logits[llm.true_id] > logits[llm.false_id]

    return res
