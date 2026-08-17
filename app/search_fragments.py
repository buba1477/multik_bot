"""Общие утилиты для формирования коротких поисковых фрагментов из названия.

Используется в publication_api.py (поиск по name) и pravo_resolver.py
(поиск по a1). Полный длинный title передавать в нечёткий поиск
порталов неэффективно — он «размывает» результат.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[а-яёa-z0-9-]+", re.IGNORECASE)
_STOP_WORDS = frozenset({
    "о", "об", "по", "в", "на", "к", "от", "с", "со",
    "из", "для", "при", "про", "и", "или", "не", "за",
    "у", "во", "обо", "а", "но", "да", "же", "ли",
    "ни", "без", "до", "над", "под", "пред", "через",
})


def build_search_fragments(title: str | None, max_per: int = 3) -> list[str]:
    """Короткие поисковые фрагменты названия — от более специфичного к менее.

    Каждый фрагмент состоит из значащих слов (без стоп-слов), длина ≤ 60
    символов. Фрагменты собираются по первым словам title, поэтому являются
    устойчивой частью названия, а не случайной выборкой.
    """
    if not title:
        return []
    words = _WORD_RE.findall(title.lower())
    words = [w for w in words if w not in _STOP_WORDS]
    fragments: list[str] = []
    for size in (4, 3, 2):
        if len(words) >= size:
            frag = " ".join(words[:size])
            if len(frag) <= 60 and frag not in fragments:
                fragments.append(frag)
                if len(fragments) >= max_per:
                    break
    return fragments
