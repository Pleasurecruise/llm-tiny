import os
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "F:/datasets/AI-Modelscope/TinyStories/data/tok4096.model"

class Tokenizer:
    """
    Intialize tokenizer load pretrained SentencePiece model，and set some special token ID。
    """
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()       # 句子开头 (BOS) 的ID
        self.eos_id: int = self.sp_model.eos_id()       # 句子结尾 (EOS) 的ID
        self.pad_id: int = self.sp_model.pad_id()       # 填充 (PAD) 的ID

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    """
    Encodes a string as a list of lexical ids. 
    You can choose whether to add sentence beginning and sentence end tags.
    """
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    """
    Decodes the list of lexical ids into strings.
    """
    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)