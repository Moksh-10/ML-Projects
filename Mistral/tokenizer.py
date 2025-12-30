from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List

class tok:
    def __init__(self, model_path:str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self):
        return self._model.vocab_size()

    @property
    def bos_id(self):
        return self._model.bos_id()

    @property
    def eos_id(self):
        return self._model.eos_id()

    @property
    def pad_id(self):
        return self._model.pad_id()

    def encode(self, s: str, bos: bool=True):
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]):
        return self._model.decode(t)