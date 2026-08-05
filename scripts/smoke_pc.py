import numpy as np
import sys
sys.path.insert(0, "src")
from aira import pc

L, d, alpha, beta = 8, 8, 1.0, 100.0
Ws = pc.make_chain(L, d, alpha, seed=0)
x, y = pc.make_io(Ws, seed=0)
A, b = pc.ab_matrices(Ws, x, y, beta)
lam_min, lam_max = pc.eig_bounds(A)
print(f"A: {A.shape}, SPD λ_min={lam_min:.5f} λ_max={lam_max:.3f} κ={lam_max/lam_min:.1f}")

s0 = pc.ff_states_vec(Ws, x)
s_eq = pc.exact_solve(A, b)
grads = pc.bp_gradient(Ws, x, y, beta)
upds = pc.pc_update(s_eq, Ws, x, y, beta)
cmin, cglob = pc.cos_align(upds, grads)
print(f"exact solve: cos_min={cmin:.6f} cos_global={cglob:.6f}")

s, wu, hist, ok = pc.solve_multigrid(A, b, s0, L, d, T_max=50, tol=1e-4,
                                     probe=lambda sv: pc.cos_align(pc.pc_update(sv, Ws, x, y, beta), grads)[1])
print(f"MG: ok={ok} wu={wu:.2f}, cos_hist={['%.3f' % h[1] for h in hist[:8]]}")

s, wu_r, hist_r, ok_r = pc.solve_richardson(A, b, s0, T_max=20000, tol=1e-4,
                                            probe=lambda sv: pc.cos_align(pc.pc_update(sv, Ws, x, y, beta), grads)[1])
print(f"Richardson: ok={ok_r} wu={wu_r:.1f}, cos_hist={['%.3f' % h[1] for h in hist_r[:8]]} ... last={hist_r[-1][1]:.4f}")

s, wu_b, ok_b = pc.solve_block_gauss_seidel(A, b, s0, zone_layers=2, T_max=200, tol=1e-4, n_layers=L, d=d)
print(f"ZoneGS(z=2): ok={ok_b} wu={wu_b:.2f}")
