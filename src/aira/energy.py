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
