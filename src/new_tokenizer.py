from src import Small_LLM_Model, get_vocab, get_inverted_vocab
from pydantic import BaseModel, model_validator, ConfigDict  # type: ignore


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = 'Qwen/Qwen3-0.6B'
    llm: Small_LLM_Model | None = None
    vocab: dict[str, int] = {}
    inv_vocab: dict[int, str] = {}
    max_len_vocab: int = 0
    fn_name_tokens: set[int] = set()
    string_tokens: set[int] = set()
    number_tokens: set[int] = set()
    integer_tokens: set[int] = set()
    true_id: int = 0
    false_id: int = 0

    @model_validator(mode='after')
    def initialize(self) -> 'LLM':
        self.llm = Small_LLM_Model(self.model)
        self.vocab = get_vocab(self.llm)
        self.inv_vocab = get_inverted_vocab(self.llm)
        self.max_len_vocab = max(len(key) for key in self.vocab.keys())
        self._cache_fn_param_tokens()
        return self

    def _cache_fn_param_tokens(self) -> None:
        for token_id, token_str in self.inv_vocab.items():
            if not token_str:
                continue

            if all(
                (c.islower() or c.isdigit() or c == '_')
                for c in token_str
            ):
                self.fn_name_tokens.add(token_id)

            if '"' not in token_str:
                self.string_tokens.add(token_id)
            if '"' in token_str:
                if token_str.endswith('"'):
                    self.string_tokens.add(token_id)

            if "'" in token_str:
                if token_str.endswith("'"):
                    if all(c in "0123456789+-.'" for c in token_str):
                        self.number_tokens.add(token_id)
                    if all(c in "0123456789+-'" for c in token_str):
                        self.integer_tokens.add(token_id)
            else:
                if all(c in "0123456789+-." for c in token_str):
                    self.number_tokens.add(token_id)

                if all(c in "0123456789+-" for c in token_str):
                    self.integer_tokens.add(token_id)

        self.true_id = self.ft_encode('true')[0]
        self.false_id = self.ft_encode('false')[0]

    def ft_encode(self, text: str) -> list[int]:
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

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return (
            self.llm.get_logits_from_input_ids(input_ids) if self.llm else []
        )
