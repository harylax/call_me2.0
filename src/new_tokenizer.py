from src import Small_LLM_Model, get_vocab, get_inverted_vocab, FunctionDef
from pydantic import BaseModel, model_validator, ConfigDict  # type: ignore
from typing import Any


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = 'Qwen/Qwen3-0.6B'
    llm: Small_LLM_Model | None = None
    vocab: dict[str, int] = {}
    inv_vocab: dict[int, str] = {}
    max_len_vocab: int = 0
    fn_name_tokens: set[int] = set()
    string_tokens: set[int] = set()
    start_number_tokens: set[int] = set()
    start_integer_tokens: set[int] = set()
    mid_number_tokens: set[int] = set()
    mid_integer_tokens: set[int] = set()
    true_id: int = 0
    false_id: int = 0

    @model_validator(mode='after')
    def initialize(self) -> 'LLM':
        self.llm = Small_LLM_Model(self.model)
        self.vocab = get_vocab(self.llm)
        self.inv_vocab = get_inverted_vocab(self.llm)
        self.max_len_vocab = max(len(key) for key in self.vocab.keys())
        self._cache_param_tokens()
        return self

    def cache_fn_name_tokens(self, functions: list[FunctionDef]) -> None:
        valid_chars: set[str] = set()
        for func in functions:
            valid_chars.update(func.name)
        for token_id, token_str in self.inv_vocab.items():
            if all(c in valid_chars for c in token_str):
                self.fn_name_tokens.add(token_id)

    def _cache_param_tokens(self) -> None:
        for token_id, token_str in self.inv_vocab.items():
            if not token_str:
                continue

            if '"' not in token_str:
                self.string_tokens.add(token_id)
            if '"' in token_str:
                if token_str.endswith('"'):
                    self.string_tokens.add(token_id)

            if token_str.count('+') > 1 or token_str.count('-') > 1:
                continue
            if '+' in token_str:
                if '-' in token_str:
                    continue
            if token_str.count('.') > 1:
                continue
            if token_str.count("'") > 1:
                continue

            if token_str.startswith('+') or token_str.startswith('-'):
                try:
                    if token_str[1] in ["'", "."]:
                        continue
                except IndexError:
                    continue
                if all(c in "0123456789+-.'" for c in token_str):
                    if "'" in token_str and not token_str.endswith("'"):
                        continue
                    if token_str.startswith('.'):
                        continue
                    self.start_number_tokens.add(token_id)

                    if '.' in token_str:
                        continue
                    self.start_integer_tokens.add(token_id)

            if token_str[0].isdigit():
                if all(c in "0123456789.'" for c in token_str):
                    if "'" in token_str and not token_str.endswith("'"):
                        continue
                    self.start_number_tokens.add(token_id)
                    if '.' in token_str:
                        continue
                    self.start_integer_tokens.add(token_id)

            if all(c in "0123456789.'" for c in token_str):
                if "'" in token_str and not token_str.endswith("'"):
                    continue
                self.mid_number_tokens.add(token_id)
                if '.' in token_str:
                    continue
                self.mid_integer_tokens.add(token_id)

        self.true_id = self.ft_encode('true')[0]
        self.false_id = self.ft_encode('false')[0]

    def ft_encode(self, text: str) -> list[int]:
        try:
            return [self.vocab[text]]
        except KeyError:
            pass
        if len(text) > 20:
            return self.llm.encode(text)[0].tolist() if self.llm else []
        token_ids: list[int] = []
        i: int = 0
        len_text: int = len(text)
        while i < len_text:
            mx: int = min(self.max_len_vocab, len_text - i)
            for j in range(i + mx, i, -1):
                sub: str = text[i:j]
                try:
                    token_id: int = self.vocab[sub]
                    token_ids.append(token_id)
                    i = j
                    break
                except KeyError:
                    continue
        return token_ids

    def ft_decode(self, token_ids: list[int] | int) -> str:
        if isinstance(token_ids, int):
            return self.inv_vocab.get(token_ids, '')
        if len(token_ids) > 20:
            return self.llm.decode(token_ids) if self.llm else ''
        return ''.join(
            self.inv_vocab.get(token_id, '')
            for token_id in token_ids
            )

    # def get_logits(self, input_ids: list[int]) -> list[float]:
    #     return (
    #         self.llm.get_logits_from_input_ids(input_ids) if self.llm else []
    #     )

    # ///////!\\\\\\\
    # For TEST
    def get_logits(
        self,
        input_ids: list[int],
        cache: Any = None,
            ) -> tuple[Any, list[float]]:
        return (
            self.llm.get_logits_from_input_ids(input_ids, cache)
            if self.llm is not None else (None, []))
