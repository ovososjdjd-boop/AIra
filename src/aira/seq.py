"""Зонно-рекуррентный носитель последовательности (M3 v0) — ZoneRNN (numpy).

Блок M2 (EXP-07, zone.py) перенесён на ось времени: состояние s_t каждого шага — узел
со своей ошибкой предсказания, а «следующим слоем» для шага t служит шаг t+1:

    E = Σ_t [ ½‖s_t − tanh(x_t Wx + s_{t−1} Wh + b)‖² ] + β·Σ_t CE(softmax(W3 s_t), y_t)

Свойства (наследуются от блока M2, фиксируются честно):
  - локальный кредит ВО ВРЕМЕНИ: градиент ошибки диффундирует на ~T итераций назад
    (трилемма FF-12 на временной топологии — документируем, не прячем);
  - солвер тот же: diag-предобусловливатель (потребители: рекуррент Wh и голова W3) ×
    BB1-шаг × сторож монотонности энергии × покоординатный сон (freeze);
  - градиенты весов локальные произведения ошибка×вход по всем шагам;
  - BP-эталон — аналитический BPTT тем же кодом, общий AdamW.
"""
from __future__ import annotations

import math

import numpy as np

from aira.zone import softmax  # noqa: F401  (переиспользуем)
from aira.zone import AdamW    # noqa: F401


class ZoneRNN:
    def __init__(self, vocab: int = 64, d_emb: int = 32, d_h: int = 96, seed: int = 42):
        self.vocab, self.d_emb, self.d_h = vocab, d_emb, d_h
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.05, (vocab, d_emb)).astype(np.float32)
        self.Wx = rng.normal(0, 1 / math.sqrt(d_emb), (d_h, d_emb)).astype(np.float32)
        self.Wh = rng.normal(0, 1 / math.sqrt(d_h), (d_h, d_h)).astype(np.float32)
        self.b = np.zeros(d_h, np.float32)
        self.W3 = rng.normal(0, 1 / math.sqrt(d_h), (vocab, d_h)).astype(np.float32)
        self.b3 = np.zeros(vocab, np.float32)

    @property
    def n_params(self) -> int:
        return sum(a.size for a in self.arrays().values())

    def arrays(self) -> dict[str, np.ndarray]:
        return {"emb": self.emb, "Wx": self.Wx, "Wh": self.Wh, "b": self.b,
                "W3": self.W3, "b3": self.b3}

    def load_arrays(self, p: dict[str, np.ndarray]) -> None:
        for k, v in p.items():
            getattr(self, k)[...] = v

    # --- свободный проход по времени ---
    def forward(self, idx: np.ndarray):
        """idx (B,L) → состояния (B,L,d_h) и логиты (B,L,V); h(0)=0."""
        B, L = idx.shape
        x = self.emb[idx]                                    # (B,L,d_emb)
        s = np.zeros((B, L, self.d_h), np.float32)
        h = np.zeros((B, self.d_h), np.float32)
        for t in range(L):
            h = np.tanh(x[:, t] @ self.Wx.T + h @ self.Wh.T + self.b)
            s[:, t] = h
        logits = s @ self.W3.T + self.b3
        return logits, s, x

    def ppl(self, idx: np.ndarray, y: np.ndarray) -> float:
        logits, _, _ = self.forward(idx)
        p = softmax(logits.reshape(-1, self.vocab))
        yy = y.reshape(-1)
        ce = -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)).mean()
        return float(np.exp(ce))

    # --- BP-BPTT эталон ---
    def bptt_grads(self, idx: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
        B, L = idx.shape
        N = B * L
        logits, s, x = self.forward(idx)
        p = softmax(logits.reshape(-1, self.vocab)).reshape(B, L, self.vocab)
        yy = y.reshape(-1)
        loss = -np.log(np.clip(p.reshape(-1, self.vocab)[np.arange(N), yy], 1e-12, 1)).mean()
        dlog = p
        dlog[np.arange(B)[:, None], np.arange(L)[None, :], y] -= 1.0   # (B,L,V)
        g = {"W3": (dlog.reshape(-1, self.vocab).T @ s.reshape(-1, self.d_h)) / N,
             "b3": dlog.mean((0, 1))}
        ds = dlog @ self.W3                                # (B,L,d_h)
        gWx = np.zeros_like(self.Wx); gWh = np.zeros_like(self.Wh)
        gb = np.zeros_like(self.b); dx = np.zeros_like(x)
        dh_next = np.zeros((B, self.d_h), np.float32)
        h_prev_of = np.concatenate([np.zeros((B, 1, self.d_h), np.float32), s[:, :-1]], 1)
        for t in reversed(range(L)):
            dt = (ds[:, t] + dh_next) * (1 - s[:, t] ** 2)
            gWx += dt.T @ x[:, t]
            gWh += dt.T @ h_prev_of[:, t]
            gb += dt.sum(0)
            dx[:, t] = dt @ self.Wx
            dh_next = dt @ self.Wh
        g["Wx"], g["Wh"], g["b"] = gWx / N, gWh / N, gb / N
        g_emb = np.zeros_like(self.emb)
        np.add.at(g_emb, idx.reshape(-1), dx.reshape(-1, self.d_emb))
        g["emb"] = g_emb / N
        return g, float(loss)

    # --- PC-релаксация состояний во времени (солвер M2) ---
    def _ener(self, s, x, y, beta):
        """Энергия E по состояниям (сторож солвера)."""
        B, L, _ = s.shape
        h_prev = np.concatenate([np.zeros((B, 1, self.d_h), np.float32), s[:, :-1]], 1)
        a = np.tanh(x @ self.Wx.T + h_prev @ self.Wh.T + self.b)
        e = s - a
        logits = s @ self.W3.T + self.b3
        p = softmax(logits.reshape(-1, self.vocab))
        ce = -np.log(np.clip(p[np.arange(B * L), y.reshape(-1)], 1e-12, 1)).mean()
        return 0.5 * float((e**2).mean()) + beta * float(ce)

    def pc_relax(self, idx: np.ndarray, y: np.ndarray, beta: float = 0.1,
                 T: int = 16, alpha: float = 1.0, freeze: float = 0.0,
                 eps: float = 0.0) -> dict:
        """Релаксация s к min E; солвер: diag-Jacobi × BB1 × сторож E × freeze."""
        B, L = idx.shape
        y_oh = np.zeros((B, L, self.vocab), np.float32)
        y_oh[np.arange(B)[:, None], np.arange(L)[None, :], y] = 1.0
        _, s, x = self.forward(idx)
        rWh = (self.Wh**2).sum(0)                            # rowsum(Wh²) по входу
        rW3 = (self.W3**2).sum(0)
        s_prev, g_prev = None, None
        E_prev, prev_resid = math.inf, math.inf
        damp, stop = 1.0, "T"
        act = np.ones(s.shape, bool)
        work, work_full, resid, t_used = 0, 0, 0.0, 0
        for t in range(T):
            h_prev = np.concatenate([np.zeros((B, 1, self.d_h), np.float32), s[:, :-1]], 1)
            a = np.tanh(x @ self.Wx.T + h_prev @ self.Wh.T + self.b)
            ap = 1 - a**2
            e = s - a
            m_next = np.zeros_like(s)
            m_next[:, :-1] = (e[:, 1:] * ap[:, 1:]) @ self.Wh   # Whᵀ(e_{t+1}⊙a'_{t+1})
            logits = s @ self.W3.T + self.b3
            dce = softmax(logits.reshape(-1, self.vocab)).reshape(B, L, self.vocab) - y_oh
            g = e - m_next + beta * (dce @ self.W3)
            D = 1.0 + ap**2 * (rWh + beta * rW3)             # потребители: Wh и голова
            pg = g / D
            resid = float(np.abs(g).max())
            if s_prev is None:
                E_prev = self._ener(s, x, y, beta)
                s_prev, g_prev = s.copy(), pg.copy()
                s[act] -= alpha * pg[act]
            else:
                ds_ = s - s_prev
                dg_ = pg - g_prev
                num = abs(float((ds_ * dg_).sum()))
                den = float((dg_ * dg_).sum()) + 1e-30
                bb = float(np.clip(num / den, 0.05, 2.0)) if num > 0 else 1.0
                E_here = self._ener(s, x, y, beta)
                if E_here > E_prev * (1 + 1e-4):
                    damp *= 0.5
                    s = s_prev
                    resid = prev_resid
                else:
                    damp = min(damp * 1.15, 1.0)
                    E_prev = E_here
                    s_prev, g_prev = s.copy(), pg.copy()
                    s[act] -= damp * bb * pg[act]
            t_used = t + 1
            if freeze:
                act = np.abs(pg) >= freeze
                work += int(act.sum()); work_full += s.size
            if eps and resid < eps:
                stop = "eps"
                break
            if t >= 8 and (prev_resid - resid) < 1e-4 and resid < 0.02:
                stop = "stalled"
                break
            prev_resid = resid
        return {"s": s, "x": x, "T_used": t_used, "resid": resid, "stop": stop,
                "work_frac": (work / work_full) if work_full else 1.0,
                "T_coord_mean": (t_used * work / work_full) if work_full else float(t_used)}

    # --- локальные градиенты весов при заданных состояниях ---
    def ep_local_grads(self, st: dict, idx: np.ndarray, y: np.ndarray,
                       beta: float) -> dict:
        B, L = idx.shape
        N = B * L
        s, x = st["s"], st["x"]
        h_prev = np.concatenate([np.zeros((B, 1, self.d_h), np.float32), s[:, :-1]], 1)
        a = np.tanh(x @ self.Wx.T + h_prev @ self.Wh.T + self.b)
        e = s - a
        t_ = e * (1 - a**2)
        logits = s @ self.W3.T + self.b3
        y_oh = np.zeros((B, L, self.vocab), np.float32)
        y_oh[np.arange(B)[:, None], np.arange(L)[None, :], y] = 1.0
        dce = softmax(logits.reshape(-1, self.vocab)).reshape(B, L, self.vocab) - y_oh
        g = {"W3": beta * (dce.reshape(-1, self.vocab).T @ s.reshape(-1, self.d_h)) / N,
             "b3": beta * dce.mean((0, 1)),
             "Wx": -(t_.reshape(-1, self.d_h).T @ x.reshape(-1, self.d_emb)) / N,
             "Wh": -(t_.reshape(-1, self.d_h).T @ h_prev.reshape(-1, self.d_h)) / N,
             "b": -t_.mean((0, 1))}
        dx = -(t_ @ self.Wx)
        g_emb = np.zeros_like(self.emb)
        np.add.at(g_emb, idx.reshape(-1), dx.reshape(-1, self.d_emb))
        g["emb"] = g_emb / N
        return g

    # --- полный шаг: однофазный или двухфазный ---
    def pc_grads(self, idx: np.ndarray, y: np.ndarray, beta: float = 0.1,
                 two_phase: bool = False, **relax_kw) -> tuple[dict, dict]:
        if not two_phase:
            st = self.pc_relax(idx, y, beta=beta, **relax_kw)
            g = self.ep_local_grads(st, idx, y, beta)
            st.pop("s"), st.pop("x")
            return g, st
        st0 = self.pc_relax(idx, y, beta=0.0, **relax_kw)
        st1 = self.pc_relax(idx, y, beta=beta, **relax_kw)
        g0 = self.ep_local_grads(st0, idx, y, 0.0)
        g1 = self.ep_local_grads(st1, idx, y, beta)
        g = {k: (g1[k] - g0[k]) / beta for k in g1}
        stats = {"T_used": st0["T_used"] + st1["T_used"],
                 "resid": max(st0["resid"], st1["resid"]),
                 "work_frac": (st0["work_frac"] + st1["work_frac"]) / 2,
                 "T_coord_mean": st0["T_coord_mean"] + st1["T_coord_mean"],
                 "stop": st1["stop"]}
        return g, stats
