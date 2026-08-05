"""Σ-δ (сигма-дельта, пороговое событийное) кодирование — THEORY §1 и §3.

Кодек: реконструкция ŝ двигается к сигналу s только событиями, когда |s − ŝ| > θ.
Свойства, проверяемые экспериментально (EXP-02):
  * max |s − ŝ| ≤ θ                          (точность по построению)
  * n_events ≤ V_T/θ + 1                      (FF-1: работа ∝ вариации, не размеру)
  * пороговое обновление сжимающей динамики h_t = λ h_{t−1} + x_t с событийным входом
    имеет дрейф ≤ θ_eff/(1−λ)                 (FF-3)
"""
from __future__ import annotations

import numpy as np


def sigma_delta_encode(x: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Кодирует 1-D сигнал пороговыми событиями.

    Возвращает (events, recon): events — массив t, где произошло событие
    (значение события = текущий уровень x[t]); recon — реконструкция ŝ(t)."""
    x = np.asarray(x, dtype=float)
    recon = np.zeros_like(x)
    level = 0.0
    events = []
    for t, v in enumerate(x):
        if abs(v - level) > theta:
            level = v
            events.append(t)
        recon[t] = level
    return np.array(events, dtype=int), recon


def total_variation(x: np.ndarray) -> float:
    """V_T = Σ|Δx| — вариация сигнала (верхний бюджет числа событий FF-1)."""
    return float(np.sum(np.abs(np.diff(x))))


def threshold_dynamics(x_recon: np.ndarray, lam: float, h0: float = 0.0) -> np.ndarray:
    """Сжимающая динамика на σ-δ-реконструированном входе: h_t = λ h_{t−1} + x_t."""
    h = np.zeros_like(x_recon, dtype=float)
    h_prev = h0
    for t, v in enumerate(x_recon):
        h_prev = lam * h_prev + v
        h[t] = h_prev
    return h
