"""Энергопрокси-счётчик (P0-зачаток) — ENERGY_MODEL.md.

Рабочие цены (пДж):
  e_DDR  = 60 /байт   e_HBM = 15 /байт   e_SRAM = 3 /байт
  e_mac_int8 = 0.5    e_mac_fp16 = 3

Прокси обучения/инференса: E ≈ (байты, прочитанные по ярусам)·e_ярус + MAC·e_mac.
Для событийных каналов: плотный шаг стоит d·b бит; событие — адрес + payload.
"""

E_DDR_PJ = 60.0
E_HBM_PJ = 15.0
E_SRAM_PJ = 3.0
E_MAC_INT8_PJ = 0.5
E_MAC_FP16_PJ = 3.0


def dense_channel_cost_pj(n_steps: int, dim: int, bits: int = 8, e_per_byte: float = E_HBM_PJ) -> float:
    """Стоимость плотной передачи dim-мерного сигнала каждый такт."""
    return n_steps * dim * bits / 8.0 * e_per_byte


def event_channel_cost_pj(n_events: int, dim_index_bits: int, payload_bits: int = 8,
                          e_per_byte: float = E_HBM_PJ) -> float:
    """Стоимость событийной передачи: каждое событие = адрес + payload."""
    bits = n_events * (dim_index_bits + payload_bits)
    return bits / 8.0 * e_per_byte


def weight_transport_pj(n_params: int, bits: int = 2, e_per_byte: float = E_HBM_PJ,
                        n_passes: float = 1.0) -> float:
    """Транспорт весов за n_passes проходов (доминирующий член обучения, FF-13)."""
    return n_params * bits / 8.0 * e_per_byte * n_passes


# ---------------------------------------------------------------------------
# Прокси v2 (M1 «стойка»): плотный трансформер, честные допущения в явном виде.
#
# MAC'и (1 MAC = 2 FLOPs):
#   fwd  ≈ N MAC/токен (N = число «матричных» параметров) + 2·L·d·ctx MAC/токен на внимание
#   bwd  = 2× fwd (градиент по входу + градиент по весам)
# Трафик весов (HBM, идеальный кэш; нижняя граница реального):
#   fwd+bwd: веса fp16 читаются 2 раза (2 Б/парам × 2)
#   оптимизатор AdamW на шаг: grad(2Б w) + w fp32(4+4) + m(4+4) + v(4+4) = 26 Б/парам
# Активации/градиенты активаций в идеале живут в SRAM; берём штрафной коэффициент
#   k_act: доля активационного трафика, пролитого в HBM (v2: k_act=0 для нижней границы).
# ---------------------------------------------------------------------------

OPTIM_BYTES_PER_PARAM = 26.0   # AdamW(fp16 grad + fp32 master/m/v), нижняя граница


def transformer_fwd_macs_per_token(n_params: int, n_layers: int, dim: int, ctx: int) -> float:
    return n_params + 2.0 * n_layers * dim * ctx


def transformer_train_token_pj(n_params: int, n_layers: int, dim: int, ctx: int,
                               tokens_per_step: float, weight_bits: int = 16,
                               e_mem: float = E_HBM_PJ, e_mac: float = E_MAC_FP16_PJ) -> float:
    """Энергия обучающего шага в пересчёте на токен (прокси, пДж/токен)."""
    fwd = transformer_fwd_macs_per_token(n_params, n_layers, dim, ctx)
    macs = 3.0 * fwd                              # fwd + 2×bwd
    e_comp = macs * e_mac
    # транспорт весов за ШАГ, делённый на токены шага
    wb = weight_bits / 8.0
    e_w = (2.0 * wb * n_params * e_mem) / tokens_per_step
    e_opt = OPTIM_BYTES_PER_PARAM * n_params * e_mem / tokens_per_step
    return e_comp + e_w + e_opt


def transformer_infer_token_pj(n_params: int, n_layers: int, dim: int, ctx: int,
                               weight_bits: int = 8, e_mem: float = E_HBM_PJ,
                               e_mac: float = E_MAC_INT8_PJ) -> float:
    """Энергия генерации токена (прокси, пДж/токен): int8, кэш KV (без пересчёта контекста)."""
    fwd = transformer_fwd_macs_per_token(n_params, n_layers, dim, ctx)
    e_comp = fwd * e_mac
    e_w = n_params * (weight_bits / 8.0) * e_mem  # веса читаются на каждый токен (decode)
    return e_comp + e_w
