# RUNBOOK H-33 — прогон-близнец (виза → старт без подготовки)

Пакет: `experiments/exp17_h33_twin.py` (руки A/B/B256/B512) +
`experiments/h33_stakes.py` (авто-вердикты S1–S6 → research/H33_STAKES.md).
Ставки заморожены: CALC05 §5 (S1–S5), CALC10_11 (S6). Химия рук: v3.2 (K1 T(β),
K2 lr×0.3 последняя треть, freeze 3e-3, шина θ=0.05, seed 42).

## Правила прогона (визовый протокол)
1. Один тяжёлый процесс в каждый момент.
2. Звенья по --max-minutes ≈ 12 (видимый прогресс каждые 300 шагов: ppl, duty, гибрид).
3. После каждого звена — короткий отчёт заказчику (ppl, duty звена, гибрид).
4. Страховка: чекпоинт на каждом отчётном шаге (включая полку руки A);
   возобновление `--resume 1` после срыва/стирания среды (данные пересобираются
   `scripts/build_corpus.py`, веса — из ckpt_h33_<tag>.npz).
5. Прогнозы уже опубликованы; изменение ставок посреди прогона запрещено.

## Порядок исполнения (суммарно ~2×60–90 мин + 2×15 мин)

Шаг 0 (калибровка темпа, 2 мин):
```
.venv/bin/python experiments/exp17_h33_twin.py --arm B --tag B_2400 --stop-at 300
```
Смотрим wall_s@300 → оцениваем мин/шаг, выставляем --max-minutes.

Рука B (эталон, зона-96 чистая v3.2): звеньями до 2400:
```
.venv/bin/python experiments/exp17_h33_twin.py --arm B --tag B_2400 --max-minutes 12 --resume 1
```
Рука A (зона-96 + полка 2D + duty-фильтр): звеньями до 2400:
```
.venv/bin/python experiments/exp17_h33_twin.py --arm A --tag A_2400 --max-minutes 12 --resume 1
```
Контроли α (младшие, те же звенья):
```
.venv/bin/python experiments/exp17_h33_twin.py --arm B256 --tag B256_2400 --max-minutes 12 --resume 1
.venv/bin/python experiments/exp17_h33_twin.py --arm B512 --tag B512_2400 --max-minutes 12 --resume 1
```

Финал: `.venv/bin/python experiments/h33_stakes.py` → таблица вердиктов
(research/H33_STAKES.md) → разбор заказчику (✓/✗ по каждой ставке отдельно).

## Что фиксируется в каждом звене (кусок из curve)
- `val_ppl` чистой зоны, `ppl_h` гибрида (рука A), `shelf_cov/acc`, `duty_link`,
  `wall_s`; `s6_prec/rec` (на финале — полные, n=20).

## Разбор отклонений
- ppl вверх ступенью на старте руки A → дрейф-маркер: пауза, диагностика β (БУМ
  болезни из EXP-14 известны: lr/2 снимает удар на второй декаде; на первой — v3.2 здорова).
- duty_link > 0.85 в первом звене → смерть S2 → стоп, разбор (полка ворует градиент:
  повысить θ или Nmin только в НОВОЙ фиксации, со сменой — это новый прогон).
- Стирание среды между звеньями → `bash scripts/setup_env.sh`, `scripts/build_corpus.py`,
  затем --resume 1 (веса и полка в чекпоинте).

## После прогона (вне этого протокола)
- Живой текст (wikitext2) — отдельный наряд (смонтирован 08.08, см. ниже).
- S6-валидатор — канонизирован в маршрутизаторе v3 (V1-ремонт, жгут b2b пройден).

---

# ЖИВОЙ НАРЯД (wikitext2) — смонтирован 08.08, пуск ТОЛЬКО по визе «пускай»

Прогнозы заморожены в H33_STAKES.md (акт L1 мостик, акт L2 главный, S1/S2/S3/
S4/S5/V3/S6' — полосы и смерти). Правила прогона те же (один процесс, звенья
≤12 мин, отчёт после каждого звена, чекпоинты, ставки не меняются посреди).

Подготовка среды после снапшота (добавлено к стандартной):
```
bash scripts/setup_env.sh && .venv/bin/python scripts/build_corpus.py \
  && .venv/bin/python scripts/build_wiki_tokenizer.py
```
(corpus_external/wikitext2 уцелевает; wiki64 воспроизводится детерминированно.)

Жгут кода v3 (ноль обучения, обязателен перед пуском и после любого патча):
```
.venv/bin/python experiments/h33_v3_burn.py   # ожидание: ✓ ЖГУТ ПРОЙДЕН
```

Акт L1 (мостик, ≈26 мин стены обеими руками):
```
.venv/bin/python experiments/exp17_h33_twin.py --arm B --src live --steps 2400 --tag liveB_2400 --max-minutes 12
.venv/bin/python experiments/exp17_h33_twin.py --arm A --src live --steps 2400 --tag liveA_2400 --max-minutes 12 \
  --sufler research/ckpts_h33/ckpt_h33_liveB_2400.npz
```
Акт L2 (главный, 76 000 шагов ≈ 9.73M символов, звеньями ≤12 мин, ≈14–16 звеньев):
```
.venv/bin/python experiments/exp17_h33_twin.py --arm B --src live --steps 76000 --tag liveB_76k --max-minutes 12 --resume 1
.venv/bin/python experiments/exp17_h33_twin.py --arm A --src live --steps 76000 --tag liveA_76k --max-minutes 12 --resume 1 \
  --sufler research/ckpts_h33/ckpt_h33_liveB_76k.npz
```
Суфлёр руки A пересматривается ПО ЗВЕНЬЯМ B (последний чекпоинт B на момент
звена A); порядок исполнения: B целиком → A целиком (суфлёр тогда финальный
B — единая эталонная точка, как в V1). Отклонения/стоп — по тем же триггерам,
плюс смерти живого наряда из H33_STAKES.md.
