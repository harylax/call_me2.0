from src import Small_LLM_Model, get_vocab, get_inverted_vocab
from pydantic import BaseModel, model_validator, ConfigDict  # type: ignore


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = 'Qwen/Qwen3-0.6B'
    llm: Small_LLM_Model | None = None
    vocab: dict[str, int] = {}
    inv_vocab: dict[int, str] = {}

    @model_validator(mode='after')
    def initialize(self) -> 'LLM':
        self.llm = Small_LLM_Model(self.model)
        self.vocab = get_vocab(self.llm)
        self.inv_vocab = get_inverted_vocab(self.llm)
        return self

    def ft_encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        i: int = 0
        while i < len(text):
            for j in range(len(text), i, -1):
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
        res: str = ''
        for token_id in token_ids:
            res += self.inv_vocab.get(token_id, '')
        return res

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return (
            self.llm.get_logits_from_input_ids(input_ids) if self.llm else []
        )
