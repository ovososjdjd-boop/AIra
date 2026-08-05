"""HDC-примитивы (кластер O, BRAIN_CUES §М): биполярные гипервектора ±1 (int8), биндинг,
бандлинг, ближайший сосед.

Коды: элемент iid Rademacher (±1) — почти ортогональны: ⟨a,b⟩/D ~ N(0, 1/D).
Биндинг (роль⊗наполнитель) — покомпонентное произведение: самообратим (a⊗a = 1),
дистрибутивен, сохраняет ортогональность.
Бандлинг — поэлементная сумма + порог до ±1 (мажоритарно).
Стоимость по прокси: MAC над ±1 = 1-битовая операция (см. ENERGY_MODEL; цена e_mac_1bit).
"""
from __future__ import annotations

import numpy as np

DIMER_DEFAULT = 2048


def make_codebook(labels: list[str], dim: int = DIMER_DEFAULT, seed: int = 0) -> dict[str, np.ndarray]:
    """Детерминированная книга кодов: label → {-1,+1}^dim (int8)."""
    rng = np.random.default_rng(seed)
    return {lab: rng.choice(np.int8([-1, 1]), size=dim) for lab in labels}


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def bundle(vecs: list[np.ndarray]) -> np.ndarray:
    s = np.sum(np.stack(vecs), axis=0)
    return np.sign(s).astype(np.int8) + (s == 0)  # нули → +1 (детерминированно)


def encode_fact(role_filler: dict[str, str], cb: dict[str, np.ndarray]) -> np.ndarray:
    """Код факта = бандлинг биндингов роль⊗наполнитель (суперпозиция слотов)."""
    return bundle([bind(cb["role:" + r], cb[f]) for r, f in role_filler.items()])


def compose_cue(partial: dict[str, str], cb: dict[str, np.ndarray]) -> np.ndarray:
    """Крючок = тот же бандлинг, но по подмножеству слотов."""
    return bundle([bind(cb["role:" + r], cb[f]) for r, f in partial.items()])


def sim(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированное сходство ⟨a,b⟩/D ∈ [-1, 1] (для int8-биполярных)."""
    return float(np.dot(a.astype(np.float64), b.astype(np.float64))) / a.size


def scan_sims(codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Сходства запроса со всеми кодами: (n,D) @ (D,) / D."""
    return codes.astype(np.float32) @ q.astype(np.float32) / codes.shape[1]
