"""Генеративный корпус M1 (v0.1) — контролируемый «мир» TinyStories-уровня.

Три слоя:
  1. stories     — шаблонные детские истории (связность, начало-цепочка-конец);
  2. facts       — пары факт/вопрос-ответ (заготовка «мир фактов» M5: one-shot память);
  3. situations  — этические мини-ситуации с меткой good/bad (заготовка батареи M7).

Всё детерминировано seed'ом, без внешних зависимостей — воспроизводимость из коробки.
Грамматика: согласование по роду существа (м/ж), падежи «встретил + р.п.», «к + д.п.»,
«живёт + предл.п. с правильным предлогом (в/на)».
"""
from __future__ import annotations

import random

# (имя, грамм. род)
NAMES = [
    ("Лёва", "м"), ("Мила", "ж"), ("Тоша", "м"), ("Ника", "ж"), ("Гоша", "м"),
    ("Ася", "ж"), ("Юра", "м"), ("Лада", "ж"), ("Сима", "ж"), ("Прохор", "м"),
]
# (им.п., р.п. после «встретил(а)», грамм. род)
CREATURES = [
    ("котёнок", "котёнка", "м"), ("лисёнок", "лисёнка", "м"), ("ёжик", "ёжика", "м"),
    ("зайчонок", "зайчонка", "м"), ("медвежонок", "медвежонка", "м"),
    ("сороконожка", "сороконожку", "ж"), ("дракончик", "дракончика", "м"),
    ("воробей", "воробья", "м"),
]
# (им.п., д.п. после «к», предлог + предл.п. для «живёт …»)
PLACES = [
    ("лес", "лесу", "в лесу"), ("берег реки", "берегу реки", "на берегу реки"),
    ("старый дуб", "старому дубу", "в старом дубе"), ("поляна", "поляне", "на поляне"),
    ("холм", "холму", "на холме"), ("пещера", "пещере", "в пещере"),
    ("деревня", "деревне", "в деревне"), ("болото", "болоту", "на болоте"),
]
OBJECTS = ["фонарик", "красный шар", "карта клада", "звонкий колокольчик", "семечко дуба",
           "вязаная шапка", "ожерелье из рябины", "пищалка сороки"]
WEATHERS = ["солнечный", "дождливый", "туманный", "снежный", "тёплый"]
FEELINGS = ["радость", "грусть", "страх", "любопытство", "гордость", "стеснение"]
MORALS_GOOD = [
    ("поделился находкой с другом", "делиться находками с друзьями — добро", "good"),
    ("вернул потерянную вещь хозяину", "возвращать чужое — честно", "good"),
    ("сказал правду, хотя было страшно", "правду говорить лучше, чем врать", "good"),
    ("помог тому, кто упал", "помогать упавшим — правильно", "good"),
]
MORALS_BAD = [
    ("спрятал чужую вещь и соврал", "прятать чужое и врать — плохо", "bad"),
    ("смеялся над тем, кто упал", "смеяться над упавшим — зло", "bad"),
    ("взял всё себе и никому не дал", "жадничать — плохо", "bad"),
]
# мужские глаголы прош. вр. → женские (для согласования в ситуациях)
_FEM = {"поделился": "поделилась", "вернул": "вернула", "сказал": "сказала",
        "помог": "помогла", "спрятал": "спрятала", "смеялся": "смеялась",
        "взял": "взяла", "соврал": "соврала"}


def _fem(text: str) -> str:
    """Мужская форма прош. вр. → женская (только для слов из _FEM)."""
    return " ".join(_FEM.get(w, w) for w in text.split())


def _genderize(verbs: tuple[str, str], g: str) -> str:
    """verbs = (форма м.р., форма ж.р.)."""
    return verbs[0] if g == "м" else verbs[1]


def make_stories(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        name, _ = rng.choice(NAMES)
        cr, cr_gen, g = rng.choice(CREATURES)          # род истории = род существа
        _, pl_dat, _ = rng.choice(PLACES)
        ob, w = rng.choice(OBJECTS), rng.choice(WEATHERS)
        feel = rng.choice(FEELINGS)
        he = _genderize(("он", "она"), g)
        kotory = _genderize(("который", "которая"), g)
        drozh = _genderize(("дрожал", "дрожала"), g)
        key_moment = rng.choice([
            f"{_genderize(('потерял', 'потеряла'), g)} {ob}",
            f"{_genderize(('нашёл', 'нашла'), g)} {ob} под кустом",
            f"{_genderize(('услышал', 'услышала'), g)} пение из кустов",
            f"{_genderize(('увидел', 'увидела'), g)} следы, ведущие к воде",
            f"{_genderize(('встретил', 'встретила'), g)} {cr_gen}, {kotory} {drozh}",
        ])
        end = rng.choice([
            f"И {cr} {_genderize(('понял', 'поняла'), g)}: {feel} бывает только у смелых. "
            f"С тех пор они с {name} были друзьями.",
            f"Вечером {name} {_genderize(('рассказал', 'рассказала'), g)} всё маме, "
            f"и мама сказала: «Молодец!». Конец.",
            f"На обратном пути опять пошёл дождь, но {name} уже не "
            f"{_genderize(('грустил', 'грустила'), g)}. Конец.",
        ])
        out.append(
            f"Жил-был {cr} по имени {name}. Однажды в {w} день {he} "
            f"{_genderize(('пошёл', 'пошла'), g)} гулять к {pl_dat}. "
            f"Там {he} {key_moment}. {name} {_genderize(('испугался', 'испугалась'), g)} "
            f"сначала, но потом {_genderize(('собрался', 'собралась'), g)} и "
            f"{_genderize(('пошёл', 'пошла'), g)} вперёд. {end}"
        )
    return out


def make_facts(n: int, seed: int = 1) -> list[dict]:
    """Факты вида ('Тоша — ёжик из леса', вопрос 'Кто такой Тоша?', ответ)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, _ = rng.choice(NAMES)
        cr, _, g = rng.choice(CREATURES)
        _, _, pl_pp = rng.choice(PLACES)
        ob = rng.choice(OBJECTS)
        kotory = _genderize(("который", "которая"), g)
        zhiv = _genderize(("живёт", "живёт"), g)
        kind = rng.choice(["кто", "где", "что"])
        who_fact = f"{name} — {cr}, {kotory} {zhiv} {pl_pp}"
        where_fact = f"{name} {zhiv} {pl_pp}"
        obj_fact = f"у {name} есть {ob}"
        if kind == "кто":
            q, a = f"Кто такой {name}?", who_fact
        elif kind == "где":
            q, a = f"Где живёт {name}?", where_fact
        else:
            q, a = f"Что есть у {name}?", obj_fact
        out.append({"id": i, "fact": a + ".", "question": q, "answer": a + "."})
    return out


def make_situations(n: int, seed: int = 2) -> list[dict]:
    """Этические мини-ситуации: текст + метка good/bad + формулировка нормы."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, g = rng.choice(NAMES)
        act, norm, label = rng.choice(MORALS_GOOD if i % 2 == 0 else MORALS_BAD)
        act = act if g == "м" else _fem(act)
        txt = f"{name} {act}. {norm[0].upper()}{norm[1:]}."
        out.append({"id": i, "text": txt, "label": label, "norm": norm})
    return out


def stats(texts: list[str]) -> dict:
    vocab = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {"n": len(texts), "vocab": len(vocab),
            "avg_words": round(sum(len(t.split()) for t in texts) / len(texts), 1)}


if __name__ == "__main__":
    st = make_stories(5, seed=7)
    for s in st:
        print("-", s)
    print(stats(make_stories(2000)))
    f = make_facts(4)
    print(f[0])
    sit = make_situations(6)
    print(sit[0], sit[1])
