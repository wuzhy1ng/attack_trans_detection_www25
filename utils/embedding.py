from functools import lru_cache
from typing import List, Dict, Iterator

from pyevmasm import evmasm
from transformers import RobertaTokenizer

from settings import HUGGING_MODEL_PATH

_tokenizer = RobertaTokenizer.from_pretrained(
    '{}/codebert-base'.format(HUGGING_MODEL_PATH)
    if HUGGING_MODEL_PATH is not None
    else 'microsoft/codebert-base'
)


def text_tokenizing(token_seq: Iterator[str]) -> List[float]:
    """
    Embed the token sequence to a vector.
    """
    text_features = _tokenizer(
        token_seq, return_tensors='pt', padding="max_length",
        truncation=True, max_length=_tokenizer.model_max_length,
    )
    return text_features['input_ids'].detach().tolist()


def opcode_embedding(opcode_seq: Iterator[str]) -> List[float]:
    """
    Embed the opcode sequence to a vector.
    """

    @lru_cache
    def _opcode2idx() -> Dict[str, int]:
        rlt = set()
        for table in evmasm.instruction_tables.values():
            for key in table.keys():
                rlt.add(table[key].name)
        rlt = sorted(list(rlt))
        return {opcode: i for i, opcode in enumerate(rlt)}

    opcode2idx = _opcode2idx()
    vec = [0.0 for _ in range(len(opcode2idx))]
    for opcode in opcode_seq:
        idx = opcode2idx.get(opcode)
        if idx is None:
            continue
        vec[idx] += 1
    return vec
