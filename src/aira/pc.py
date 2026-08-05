"""Равновесное обучение (предиктивное кодирование) — ядро EXP-01 (THEORY §11).

Энергия цепочки (Π = I, мягкий зажим выхода силы β):

    E(s) = Σ_l ½‖s_l − W_l f(s_{l−1})‖²  +  β/2‖s_L − y*‖²,     s_0 = x (фиксирован)

Обучающее правило в равновесии (строго локально):  ΔW_l = η_w · ε_l · f(s_{l−1})ᵀ.

Солверы равновесия:
  * Richardson        — глобальный градиентный спуск (база);
  * Jacobi            — диагональный предобуславливатель (H-14);
  * block GS по зонам — зонная релаксация по глубине (THEORY §12 Т1);
  * Multigrid V-цикл  — иерархия по глубине: грубая сетка несёт дальний кредит (H-18).

Рабочая единица WU = одна полная операция уровня fine-размера (мат-век A·s);
уровень размера m стоит m/(Ld). Честный прокси «времени» релаксации.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- сеть

def make_chain(L: int, d: int, alpha: float, seed: int, kind: str = "linear"):
    """Веса W_l = alpha · Q_l (Q ортогональная) — контролируемый транспорт α."""
    rng = np.random.default_rng(seed)
    Ws = []
    for _ in range(L):
        M = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(M)
        Ws.append(alpha * Q)
    return Ws


def f_act(s, kind):
    return np.tanh(s) if kind == "tanh" else s


def f_act_deriv(s, kind):
    return 1.0 - np.tanh(s) ** 2 if kind == "tanh" else np.ones_like(s)


def forward_pass(Ws, x, kind="linear"):
    s = [x]
    for W in Ws:
        s.append(W @ f_act(s[-1], kind))
    return s


def make_io(Ws, seed, sigma=0.5, kind="linear"):
    """Вход + цель: y = чистый выход + шум (нетривиальный, но ограниченный кредит)."""
    rng = np.random.default_rng(seed + 777)
    x = rng.standard_normal(Ws[0].shape[0])
    y = forward_pass(Ws, x, kind)[-1] + sigma * rng.standard_normal(Ws[0].shape[0])
    return x, y


# ------------------------------------------------------- линейный случай: A, b

def ab_matrices(Ws, x, y, beta):
    """E(s) = ½ sᵀAs − bᵀs + const для линейной цепочки; переменные s_1..s_L."""
    L, d = len(Ws), Ws[0].shape[0]
    A = np.zeros((L * d, L * d))
    b = np.zeros(L * d)
    for l in range(1, L + 1):
        i = (l - 1) * d
        A[i:i + d, i:i + d] += np.eye(d)
        if l < L:
            W1 = Ws[l]                                    # W_{l+1}
            A[i:i + d, i:i + d] += W1.T @ W1
            A[i:i + d, i + d:i + 2 * d] -= W1.T
            A[i + d:i + 2 * d, i:i + d] -= W1
        else:
            A[i:i + d, i:i + d] += beta * np.eye(d)
            b[i:i + d] += beta * y
    b[:d] += Ws[0] @ x
    return A, b


def exact_solve(A, b):
    return np.linalg.solve(A, b)


def eig_bounds(A):
    ev = np.linalg.eigvalsh(A)
    return float(ev[0]), float(ev[-1])


# ------------------------------------------------------- BP-референс и апдейты

def bp_gradient(Ws, x, y, beta, kind="linear"):
    """Точный градиент лосса β/2‖net(x)−y‖² (в точке feedforward-прохода)."""
    s = forward_pass(Ws, x, kind)
    L = len(Ws)
    delta = beta * (s[L] - y)
    grads = [None] * L
    for l in range(L, 0, -1):
        grads[l - 1] = np.outer(delta, f_act(s[l - 1], kind))
        if l > 1:
            delta = f_act_deriv(s[l - 1], kind) * (Ws[l - 1].T @ delta)
    return grads


def pc_update(s_vec, Ws, x, y, beta, kind="linear"):
    """Градиент энергии по весам в равновесии: ∂E/∂W_l = −ε_l f(s_{l−1})ᵀ.

    Возвращаем именно ∂E/∂W (спуск = минус это), тогда знак согласован с
    BP-градиентом ∂ℓ/∂W: в равновесии −ε_l = δ_l (BP-дельта)."""
    L, d = len(Ws), Ws[0].shape[0]
    s_full = [x] + [s_vec[(l - 1) * d:l * d] for l in range(1, L + 1)]
    upds = []
    for l in range(1, L + 1):
        eps = s_full[l] - Ws[l - 1] @ f_act(s_full[l - 1], kind)
        upds.append(np.outer(-eps, f_act(s_full[l - 1], kind)))
    return upds


def ff_states_vec(Ws, x, kind="linear"):
    """Feedforward-инициализация состояний как вектор (стандартная стартовая точка PC)."""
    s = forward_pass(Ws, x, kind)
    return np.concatenate(s[1:])


def cos_align(upds, grads):
    """Косинусное выравнивание PC-апдейта с BP-градиентом: минимум по слоям + глобал."""
    cos_l = []
    all_u, all_g = [], []
    for u, g in zip(upds, grads):
        nu, ng = np.linalg.norm(u), np.linalg.norm(g)
        cos_l.append(float(np.sum(u * g) / (nu * ng + 1e-30)))
        all_u.append(u.ravel())
        all_g.append(g.ravel())
    u = np.concatenate(all_u)
    g = np.concatenate(all_g)
    glob = float(u @ g / (np.linalg.norm(u) * np.linalg.norm(g) + 1e-30))
    return min(cos_l), glob


# ------------------------------------------------------- солверы (линейная E)

def solve_richardson(A, b, s0, T_max, tol, probe=None, probe_every=25):
    """s ← s + η(b − A s), η = 1.5/λ_max. Возвращает (s, WU, hist, сошёлся)."""
    _, lam_max = eig_bounds(A)
    eta = 1.5 / lam_max
    s = s0.copy()
    r0 = np.linalg.norm(b - A @ s0) + 1e-30
    hist = []
    for t in range(T_max):
        r = b - A @ s
        if probe is not None and t % probe_every == 0:
            hist.append((float(t), probe(s)))
        if np.linalg.norm(r) / r0 < tol:
            return s, float(t), hist, True
        s += eta * r
    return s, float(T_max), hist, False


def solve_jacobi(A, b, s0, T_max, tol, omega=1.0, probe=None, probe_every=25):
    """Предобусловленный шаг: s ← s + ω·D⁻¹(b − A s), D = diag(A) (H-14)."""
    D = np.diag(A)
    s = s0.copy()
    r0 = np.linalg.norm(b - A @ s0) + 1e-30
    hist = []
    for t in range(T_max):
        r = b - A @ s
        if probe is not None and t % probe_every == 0:
            hist.append((float(t), probe(s)))
        if np.linalg.norm(r) / r0 < tol:
            return s, float(t), hist, True
        s += omega * r / D
    return s, float(T_max), hist, False


def solve_block_gauss_seidel(A, b, s0, zone_layers, T_max, tol, n_layers, d):
    """Зонная релаксация: глубина бьётся на зоны по zone_layers слоёв; каждая зона
    за визит решается точно (мелкая — дёшево, THEORY §12 Т1), зоны — последовательно
    (Гаусс–Зейдель). WU: точное решение зоны z слоёв ~ (z³d³/3)/(3Ld²) — плотная
    факторизация против мат-веков; визит зоны дешевле глобального шага при z ≪ L."""
    L = n_layers
    zones = []
    start = 0
    while start < L:
        zones.append(range(start, min(start + zone_layers, L)))
        start += zone_layers
    s = s0.copy()
    r0 = np.linalg.norm(b - A @ s0) + 1e-30
    wu = 0.0
    for _ in range(T_max):
        for z in zones:
            idx = np.concatenate([np.arange(l * d, (l + 1) * d) for l in z])
            Azz = A[np.ix_(idx, idx)]
            rhs = b[idx] - A[np.ix_(idx, np.setdiff1d(np.arange(L * d), idx))] @ s[np.setdiff1d(np.arange(L * d), idx)]
            s[idx] = np.linalg.solve(Azz, rhs)
            zdepth = len(z)
            wu += (zdepth ** 3 * d ** 3 / 3.0) / (3 * L * d * d)
        r = b - A @ s
        if np.linalg.norm(r) / r0 < tol:
            return s, wu, True
    return s, wu, False


# ------------------------------------------------------- Multigrid по глубине

def _prolongation_matrix(L, d):
    """P: coarse (Lc·d) → fine (L·d), Lc=⌈L/2⌉; слой m копируется в 2m-1, 2m."""
    Lc = L // 2 + L % 2
    P = np.zeros((L * d, Lc * d))
    for m in range(Lc):
        rows = [2 * m]
        if 2 * m + 1 < L:
            rows.append(2 * m + 1)
        for r in rows:
            P[r * d:(r + 1) * d, m * d:(m + 1) * d] = np.eye(d)
    return P, Lc


def solve_multigrid(A, b, s0, n_layers, d, T_max, tol, probe=None,
                    nu1=2, nu2=2, omega=0.6):
    """V-циклы по глубине (иерархия «грубый кредит» через скелет, H-18)."""
    # --- строим иерархию уровней: fine → coarse по парам слоёв
    levels = []
    A_l, L_l = A, n_layers
    while True:
        levels.append({"A": A_l, "D": np.diag(A_l)})
        if L_l <= 4:
            break
        P, Lc = _prolongation_matrix(L_l, d)
        levels[-1]["P"] = P
        levels[-1]["R"] = 0.5 * P.T                    # полновзвешенная рестрикция
        A_l = levels[-1]["R"] @ A_l @ P                # Галеркин: A_coarse = R A P
        L_l = Lc

    s = s0.copy()
    r0 = np.linalg.norm(b - A @ s0) + 1e-30
    hist = []
    wu_total = 0.0
    for _ in range(T_max):
        if probe is not None:
            hist.append((wu_total, probe(s)))
        if np.linalg.norm(b - A @ s) / r0 < tol:
            return s, wu_total, hist, True
        s, wu = _vcycle(levels, 0, b, s, nu1, nu2, omega)
        wu_total += wu
    return s, wu_total, hist, False


def _vcycle(levels, lev, bvec, xvec, nu1, nu2, omega):
    """Один V-цикл на уровне lev: ν1 сглаживаний → грубая коррекция → ν2.

    Возвращает (x, wu): x — обновлённое состояние, wu — работа в единицах
    fine-уровня (уровень размера m стоит m/n_fine)."""
    A_l, D_l = levels[lev]["A"], levels[lev]["D"]
    cost = A_l.shape[0] / levels[0]["A"].shape[0]
    wu = 0.0
    for _ in range(nu1):                               # пред-сглаживание
        xvec = xvec + omega * (bvec - A_l @ xvec) / D_l
        wu += cost
    if lev < len(levels) - 1:                          # грубая коррекция
        r_c = levels[lev]["R"] @ (bvec - A_l @ xvec)
        e_c, wu_c = _vcycle(levels, lev + 1, r_c, np.zeros_like(r_c), nu1, nu2, omega)
        xvec = xvec + levels[lev]["P"] @ e_c
        wu += wu_c
    else:                                              # мелкая сетка: точно и дёшево
        xvec = np.linalg.solve(A_l, bvec)
        wu += cost * 0.1
    for _ in range(nu2):                               # досглаживание
        xvec = xvec + omega * (bvec - A_l @ xvec) / D_l
        wu += cost
    return xvec, wu
