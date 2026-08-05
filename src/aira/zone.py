"""Зонно-событийный учебный блок M2 (THEORY §12 + спецификация из выводов EXP-04).

Ядро — numpy (без torch): честная стойка, где BP-тоже посчитан аналитически тем же
кодом операций, а оптимизатор AdamW общий. Модель — CharMLP в точности как EXP-04:
окно ctx символов → emb(32) → tanh(256) → tanh(256) → логиты V.

Релаксатор состояний (к равновесию энергии E, BP≡PC-мост FF-11):
    E = ½‖s1−f1‖² + ½‖s2−f2‖² + β·CE(softmax(W3 s2), y)
  - method="euler":  s ← s − α·∇E                      (наивный, как EXP-04)
  - method="jacobi": s ← s − α·(D)⁻¹∇E, диагональный предобуславливатель зоны
        D1 = 1 + (f1')² ⊙ rowsum(W2²)          (кривизна от слоя-«потребителя»)
        D2 = 1 + β·rowsum(W3²)                 (верхняя граница кривизны CE)
  - method="bb":     jacobi + спектральный BB1-шаг на зону (лечит предельный цикл,
        обнаруженный в EXP-07) + сторож монотонности энергии: рост E ⇒ откат шага и
        затухание ×½ (цена — одна оценка E на итерацию, ~+1 fwd; честно в отчёте).
    D вычислим ЛОКАЛЬНО в зоне (свои веса, свои точки) — EXP-04 показал, что наивный
    Эйлер при росте ‖W‖ входит в предел осцилляций; D сам гасит шаг на жёстких осях.
  - стопы: абсолютный ‖∇E‖∞ < eps ИЛИ относительный «stalled» (остаток не падает
    ≥1e-3 за итерацию при resid<0.05 после t≥4) — в предельном цикле Эйлера
    абсолютный eps не срабатывает никогда (EXP-07, S3-первопрогон).
  - bus: BusSigmaDelta на состояние s1 (зона1→зона2) и на latch-сообщение
    m2 = W2ᵀ(e2⊙f2') (зона2→зона1) — межзонная σ-δ шина (H-16): передаются только
    события выше порога, координаты без событий на приёмнике держат старый уровень.

Оценщики градиентов весов (оба — локальные произведения ошибка×вход, FF-11):
  - однофазный (β>0): как EXP-04 (смещён при сильном β);
  - двухфазный: (G(β) − G(0))/β по релаксированным состояниям двух фаз.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# σ-δ шина (векторная, потоковая: уровень сохраняется между вызовами)
# ---------------------------------------------------------------------------
class BusSigmaDelta:
    """Векторный σ-δ кодек: recon двигается только событиями |new−level| > θ.

    Стоимость события = адрес (addr_bits) + payload (payload_bits, float16-уровень).
    Форма сигнала — (B, d); события считаются покоординатно по всему батчу.
    """

    def __init__(self, shape: tuple[int, int], theta: float = 0.0,
                 addr_bits: int = 8, payload_bits: int = 16):
        self.theta = float(theta)
        self.level = np.zeros(shape, dtype=np.float32)
        self.addr_bits, self.payload_bits = addr_bits, payload_bits
        self.n_events, self.n_sends = 0, 0

    def send(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        self.n_sends += 1
        if self.theta <= 0:
            self.level = x.astype(np.float32, copy=True)
            self.n_events += x.size
            return self.level, x.size
        mask = np.abs(x - self.level) > self.theta
        self.level[mask] = x[mask]
        n = int(mask.sum())
        self.n_events += n
        return self.level, n

    def traffic_bits(self) -> int:
        return self.n_events * (self.addr_bits + self.payload_bits)


# ---------------------------------------------------------------------------
# CharMLP numpy (идентично EXP-04)
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CharMLP:
    def __init__(self, vocab: int = 64, ctx: int = 8, d_emb: int = 32, d_hid: int = 256,
                 seed: int = 42):
        self.vocab, self.ctx, self.d_emb, self.d_hid = vocab, ctx, d_emb, d_hid
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.05, (vocab, d_emb)).astype(np.float32)
        self.W1 = rng.normal(0, 1 / math.sqrt(ctx * d_emb), (d_hid, ctx * d_emb)).astype(np.float32)
        self.b1 = np.zeros(d_hid, np.float32)
        self.W2 = rng.normal(0, 1 / math.sqrt(d_hid), (d_hid, d_hid)).astype(np.float32)
        self.b2 = np.zeros(d_hid, np.float32)
        self.W3 = rng.normal(0, 1 / math.sqrt(d_hid), (vocab, d_hid)).astype(np.float32)
        self.b3 = np.zeros(vocab, np.float32)

    @property
    def n_params(self) -> int:
        return sum(a.size for a in self.arrays().values())

    def arrays(self) -> dict[str, np.ndarray]:
        return {"emb": self.emb, "W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def load_arrays(self, p: dict[str, np.ndarray]) -> None:
        for k, v in p.items():
            getattr(self, k)[...] = v

    def clone_params(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.arrays().items()}

    # --- прямой проход ---
    def feats(self, idx: np.ndarray):
        x = self.emb[idx].reshape(len(idx), -1)
        f1 = np.tanh(x @ self.W1.T + self.b1)
        return x, f1

    def forward(self, idx: np.ndarray):
        x, f1 = self.feats(idx)
        f2 = np.tanh(f1 @ self.W2.T + self.b2)
        logits = f2 @ self.W3.T + self.b3
        return logits, (x, f1, f2)

    def loss_ppl(self, idx: np.ndarray, y: np.ndarray) -> float:
        logits, _ = self.forward(idx)
        p = softmax(logits)
        ce = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
        return float(np.exp(ce))

    # --- BP-градиенты (аналитические, эталон) ---
    def bp_grads(self, idx: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        B = len(idx)
        logits, (x, f1, f2) = self.forward(idx)
        p = softmax(logits)
        loss = -np.log(np.clip(p[np.arange(B), y], 1e-12, 1)).mean()
        dlog = p
        dlog[np.arange(B), y] -= 1.0                       # (B,V) ∂CE/∂logits
        g = {"W3": (dlog.T @ f2) / B, "b3": dlog.mean(0)}
        dh2 = dlog @ self.W3
        dz2 = dh2 * (1 - f2**2)
        g["W2"] = (dz2.T @ f1) / B
        g["b2"] = dz2.mean(0)
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (1 - f1**2)
        g["W1"] = (dz1.T @ x) / B
        g["b1"] = dz1.mean(0)
        dx = (dz1 @ self.W1).reshape(B, self.ctx, self.d_emb)
        g_emb = np.zeros_like(self.emb)
        np.add.at(g_emb, idx.reshape(-1), dx.reshape(-1, self.d_emb))
        g["emb"] = g_emb / B
        return g, float(loss)

    # --- PC-релаксация состояний (зонная механика M2) ---
    def _state_energy(self, x, s1, s1_bus, s2, y_oh, beta):
        f1 = np.tanh(x @ self.W1.T + self.b1)
        e1 = s1 - f1
        f2 = np.tanh(s1_bus @ self.W2.T + self.b2)
        e2 = s2 - f2
        logits = s2 @ self.W3.T + self.b3
        p = softmax(logits)
        ce = -np.log(np.clip(p[y_oh == 1], 1e-12, 1)).mean()
        return 0.5 * float((e1**2).mean()) + 0.5 * float((e2**2).mean()) + beta * float(ce)

    def pc_relax(self, idx: np.ndarray, y: np.ndarray, beta: float = 0.1,
                 T: int = 32, alpha: float = 0.3, method: str = "euler",
                 eps: float = 0.0, freeze: float = 0.0,
                 bus12: BusSigmaDelta | None = None,
                 bus21: BusSigmaDelta | None = None) -> dict:
        """Релаксация (s1,s2) к min E по состояниям; веса заморожены.

        freeze>0 — СОБЫТИЙНЫЙ режим: координата, у которой |предобусловленный
        градиент| < freeze, «засыпает» — не обновляется, пока её градиент снова не
        станет большим (реактивация проверяется каждую итерацию). numpy считает
        плотно (честный симулятор), экономия считается как доля координато-шагов,
        которые пропустил бы событийный рантайм. Шина при freeze получает события
        бесплатно: неподвижная координата события не порождает.
        """
        B = len(idx)
        y_oh = np.zeros((B, self.vocab), np.float32)
        y_oh[np.arange(B), y] = 1.0
        x = self.emb[idx].reshape(B, -1)
        s1 = np.tanh(x @ self.W1.T + self.b1)               # инициализация предсказаниями
        b12 = bus12 if bus12 is not None else BusSigmaDelta(s1.shape, 0.0)
        s2 = np.tanh(b12.send(s1)[0] @ self.W2.T + self.b2)
        b21 = bus21 if bus21 is not None else BusSigmaDelta(s1.shape, 0.0)

        rW2 = (self.W2**2).sum(0)                            # rowsum(W2²) по входному индексу
        rW3 = (self.W3**2).sum(0)
        t_used, resid, quiet, stop = 0, 0.0, 0.0, "T"
        events = s1.size
        s1p = s2p = g1p = g2p = None
        damp = np.ones(2)
        E_prev = math.inf
        prev_resid = math.inf
        act1 = np.ones(s1.shape, bool)                       # событийные маски
        act2 = np.ones(s2.shape, bool)
        work, work_full = 0, 0
        for t in range(T):
            f1 = np.tanh(x @ self.W1.T + self.b1)
            f1p = 1 - f1**2
            s1_sent, n_ev = b12.send(s1)
            f2 = np.tanh(s1_sent @ self.W2.T + self.b2)
            f2p = 1 - f2**2
            e2 = s2 - f2
            m2, n_ev2 = b21.send((e2 * f2p) @ self.W2)       # W2ᵀ(e2⊙f2') из зоны2
            events += n_ev + n_ev2
            logits = s2 @ self.W3.T + self.b3
            dce = softmax(logits) - y_oh
            g1 = (s1 - f1) - m2
            g2 = e2 + beta * (dce @ self.W3)
            resid = float(max(np.abs(g1).max(), np.abs(g2).max()))
            d1 = 1.0 + f1p**2 * rW2
            d2 = np.broadcast_to(1.0 + beta * rW3, g2.shape)
            if method == "bb":
                pg1, pg2 = g1 / d1, g2 / d2
                if s1p is None:                              # первый шаг — jacobi
                    g1p, g2p = pg1, pg2
                    E_prev = self._state_energy(x, s1, s1_sent, s2, y_oh, beta)
                    s1p, s2p = s1.copy(), s2.copy()
                    s1[act1] -= alpha * pg1[act1]
                    s2[act2] -= alpha * pg2[act2]
                else:
                    bb = []
                    for ds, dg in ((s1 - s1p, pg1 - g1p), (s2 - s2p, pg2 - g2p)):
                        num = abs(float((ds * dg).sum()))
                        den = float((dg * dg).sum()) + 1e-30
                        bb.append(float(np.clip(num / den, 0.05, 2.0)) if num > 0 else 1.0)
                    E_here = self._state_energy(x, s1, s1_sent, s2, y_oh, beta)
                    if E_here > E_prev * (1 + 1e-4):         # сторож: рост E
                        damp *= 0.5                          # — откат и затухание
                        s1, s2 = s1p, s2p
                        resid = prev_resid
                    else:
                        damp = np.minimum(damp * 1.15, 1.0)
                        E_prev = E_here
                        s1p, s2p = s1.copy(), s2.copy()
                        g1p, g2p = pg1, pg2
                        s1[act1] -= damp[0] * bb[0] * pg1[act1]
                        s2[act2] -= damp[1] * bb[1] * pg2[act2]
            elif method == "jacobi":
                s1[act1] -= alpha * g1[act1] / d1[act1]
                s2[act2] -= alpha * g2[act2] / d2[act2]
            else:
                s1[act1] -= alpha * g1[act1]
                s2[act2] -= alpha * g2[act2]
            if freeze:
                ref1, ref2 = (g1 / d1, g2 / d2) if method in ("bb", "jacobi") else (g1, g2)
                act1 = np.abs(ref1) >= freeze                # заснуть/проснуться свежим градиентом
                act2 = np.abs(ref2) >= freeze
                work += int(act1.sum() + act2.sum())
                work_full += 2 * s1.size
            t_used = t + 1
            if t >= 1:
                quiet = float((np.abs(g1) < eps).mean()) if eps else 0.0
            if eps and resid < eps:
                stop = "eps"
                break
            if t >= 4 and (prev_resid - resid) < 1e-3 and resid < 0.05:
                stop = "stalled"
                break
            prev_resid = resid
        return {"x": x, "s1": s1, "s2": s2, "y_oh": y_oh, "s1_bus": b12.level,
                "T_used": t_used, "resid": resid, "quiet_frac": quiet,
                "bus_events": events, "stop": stop,
                "work_frac": (work / work_full) if work_full else 1.0,
                "T_coord_mean": (t_used * work / work_full) if work_full else float(t_used)}

    # --- локальные градиенты весов при заданных состояниях ---
    def local_grads(self, st: dict, idx: np.ndarray, beta: float,
                    s1_for_w2: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Однофазный оценщик: ∂E/∂W в равновесии (локальные произведения)."""
        B = len(idx)
        x, s1, s2, y_oh = st["x"], st["s1"], st["s2"], st["y_oh"]
        s1_w2 = s1 if s1_for_w2 is None else s1_for_w2
        f1 = np.tanh(x @ self.W1.T + self.b1)
        f1p = 1 - f1**2
        f2 = np.tanh(s1_w2 @ self.W2.T + self.b2)
        f2p = 1 - f2**2
        e1, e2 = s1 - f1, s2 - f2
        logits = s2 @ self.W3.T + self.b3
        dce = softmax(logits) - y_oh
        t1, t2 = e1 * f1p, e2 * f2p
        g = {"W1": -(t1.T @ x) / B, "b1": -t1.mean(0),
             "W2": -(t2.T @ s1_w2) / B, "b2": -t2.mean(0),
             "W3": beta * (dce.T @ s2) / B, "b3": beta * dce.mean(0)}
        dx = (t1 @ self.W1).reshape(B, self.ctx, self.d_emb)
        g_emb = np.zeros_like(self.emb)
        np.add.at(g_emb, idx.reshape(-1), dx.reshape(-1, self.d_emb))
        g["emb"] = -g_emb / B
        return g

    def pc_grads(self, idx: np.ndarray, y: np.ndarray, beta: float = 0.1,
                 two_phase: bool = False, **relax_kw) -> tuple[dict[str, np.ndarray], dict]:
        """Полный PC-шаг: одно- или двухфазный оценщик + статистика релаксации."""
        if not two_phase:
            st = self.pc_relax(idx, y, beta=beta, **relax_kw)
            bus12 = relax_kw.get("bus12")
            g = self.local_grads(st, idx, beta,
                                 s1_for_w2=None if bus12 is None else bus12.level)
            st.pop("y_oh"), st.pop("x")
            return g, st
        st0 = self.pc_relax(idx, y, beta=0.0, **relax_kw)
        st1 = self.pc_relax(idx, y, beta=beta, **relax_kw)
        g0 = self.local_grads(st0, idx, 0.0)
        g1 = self.local_grads(st1, idx, beta)
        g = {k: (g1[k] - g0[k]) / beta for k in g1}
        stats = {"T_used": st0["T_used"] + st1["T_used"],
                 "resid": max(st0["resid"], st1["resid"]), "quiet_frac": st1["quiet_frac"]}
        for k in ("x", "y_oh"):
            st0.pop(k, None); st1.pop(k, None)
        return g, stats


# ---------------------------------------------------------------------------
# AdamW (numpy, общий для обеих рук)
# ---------------------------------------------------------------------------
class AdamW:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 wd: float = 0.01):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, *betas, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in self.p.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1**self.t)
            vh = self.v[k] / (1 - self.b2**self.t)
            p -= self.lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * p)


# ---------------------------------------------------------------------------
def cos_sim(a: dict[str, np.ndarray], b: dict[str, np.ndarray],
            keys: list[str] | None = None) -> float:
    """Косинус между двумя наборами градиентов (по векторизации)."""
    keys = keys or list(a.keys())
    va = np.concatenate([a[k].ravel() for k in keys])
    vb = np.concatenate([b[k].ravel() for k in keys])
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-30))
