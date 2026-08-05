"""Посимвольный токенизатор M1 (стартовый уровень; BPE-16k — при переходе к реальным объёмам).

Детерминированный: словарь строится из обучающего корпуса (отсортированный набор символов),
специальные токены фиксированы. Сохраняется в JSON — идентичная воспроизводимость стойки.

Спецтокены:
  <pad>=0  <bos>=1  <eos>=2  <unk>=3   (символы документов начинаются с 4)
"""
from __future__ import annotations

import json
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)                       # порядок = id - 4
        self.stoi = {c: i + len(SPECIALS) for i, c in enumerate(self.chars)}
        self.itos = {i + len(SPECIALS): c for i, c in enumerate(self.chars)}

    # --- построение ---
    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({c for t in texts for c in t})
        return cls(chars)

    # --- API ---
    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(SPECIALS)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in text]
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        inv_specials = {v: k for k, v in SPECIALS.items()}
        out = []
        for i in ids:
            if i in self.itos:
                out.append(self.itos[i])
            elif not skip_special:
                out.append(inv_specials.get(i, "?"))
        return "".join(out)

    # --- сохранение/загрузка ---
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "char-v1", "chars": self.chars},
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        assert obj["kind"] == "char-v1", obj["kind"]
        return cls(obj["chars"])


if __name__ == "__main__":
    tok = CharTokenizer.from_texts(["Привет, мир!", "Мама мыла раму."])
    ids = tok.encode("Мама мыла", add_bos=True, add_eos=True)
    back = tok.decode(ids)
    print("vocab:", tok.vocab_size, "ids:", ids[:8], "…")
    print("roundtrip ok:", back == "Мама мыла")
    print("unk symbol:", tok.encode("😀"), "->", repr(tok.decode([3], skip_special=False)))
