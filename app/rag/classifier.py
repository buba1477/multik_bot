"""LLM-классификатор запросов: EXACT (0.5/0.5) vs USER (1.0/0.0).

Определяет, содержит ли запрос точные реквизиты / конкретную величину,
чтобы применить гибридные веса RRF с большим весом BM25 (0.5/0.5),
улучшая поиск по узким терминологическим запросам.

Единый источник правды по весам — словарь WEIGHTS. Режим (exact/user)
определяет LLM; маппинг режима в веса делает get_weights().
engine_rag.py читает режим через classify_mode() и маппит его в веса.

Архитектура LLM-first:
1) strong_exact — цифры ИЛИ сильные реквизиты (статья, ст., ФЗ, указ,
   постановление, %, процент) → EXACT без вызова LLM;
2) llm — всё остальное решает LLM-классификатор (_CLASSIFIER_PROMPT);
3) guard_fallback — эвристика _guard_fallback() ТОЛЬКО при непарсируемом
   ответе LLM или недоступности бэкенда.

Backend: ollama | gigachat (env CLASSIFIER_BACKEND, по умолч. ollama).
Кэш: _mode_cache — глобальный словарь модуля (ключ — нормализованный запрос).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Dict, Literal, Tuple

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# =========================================================
# Конфигурация из окружения
# =========================================================
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "ollama")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")
OLLAMA_MODEL = os.getenv("CLASSIFIER_MODEL", "yagpt5_fns:latest")

# GigaChat
GIGACHAT_KEY = os.getenv("API_KEY_GIGACHAT", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")

# =========================================================
# Режим и веса (единый источник правды)
# =========================================================
Mode = Literal["exact", "user"]

WEIGHTS: Dict[Mode, Tuple[float, float]] = {
    "exact": (0.5, 0.5),   # точные реквизиты: больше BM25
    "user": (1.0, 0.0),    # смысловые запросы: чистый dense (Вариант B, анблинг 06.10.2026)
}
DEFAULT_MODE: Mode = "user"


def get_weights(mode: Mode) -> Tuple[float, float]:
    """Единственная точка маппинга режима запроса в RRF-веса."""
    return WEIGHTS[mode]


# =========================================================
# Backend: Ollama
# =========================================================
def _ollama_generate(prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
    """Генерация через Ollama (urllib, без внешних зависимостей)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 512,
        },
    }
    req = urllib.request.Request(
        OLLAMA_HOST.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


# =========================================================
# Backend: GigaChat
# =========================================================
def _gigachat_generate(prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
    """Генерация через GigaChat SDK (gigachat.GigaChat)."""
    if not GIGACHAT_KEY:
        raise ValueError("API_KEY_GIGACHAT не задан")
    from gigachat import GigaChat as GigaChatSDK

    payload = {
        "model": GIGACHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with GigaChatSDK(credentials=GIGACHAT_KEY) as giga:
        response = giga.chat(payload)
        return response.choices[0].message.content.strip()


def _generate(prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
    """Выбор бэкенда по CLASSIFIER_BACKEND."""
    if CLASSIFIER_BACKEND == "gigachat":
        return _gigachat_generate(prompt, max_tokens, temperature)
    return _ollama_generate(prompt, max_tokens, temperature)


# =========================================================
# Публичный API
# =========================================================

# =========================================================
# Промпт LLM-классификатора
# =========================================================
_CLASSIFIER_PROMPT = (
    "Ты — маршрутизатор запросов к базе НПА ФНС России.\n\n"
    "Верни ровно ОДНО слово: EXACT или USER.\n\n"
    "ПРАВИЛО ВЫБОРА:\n"
    "- Верни EXACT, если запрос содержит точные реквизиты (номера статей, законов, указов) ИЛИ конкретные числовые критерии (проценты, стаж, сроки, суммы).\n"
    "- Верни EXACT, если запрос содержит слова график, круговая диаграмма, столбчатая диаграмма, линейный график.\n"
    "- Верни EXACT, если запрос спрашивает \"в каких случаях\", \"при каких условиях\", \"вправе ли\", \"обязан ли\", \"может ли\", \"что делать\" и касается норм, запретов, гарантий, ограничений — это EXACT, даже если в нём нет цифр и реквизитов.\n"
    "- Верни USER, если запрос является текстовым вопросом (включая тесты и квизы без цифр), описывает ситуацию, понятия (коррупция, увольнение) своими словами. Здесь важен смысл, а не цифры.\n\n"
    "Примеры EXACT (реквизиты и числовые критерии):\n"
    "\"статья 346.14 НК РФ\"\n"
    "\"Размер надбавки при стаже свыше 15 лет составляет: ? 15 процентов 20 процентов\"\n"
    "\"Какой штраф по статье 122 при умысле?\"\n\n"
    "Примеры EXACT (нормативные вопросы без цифр):\n"
    "\"В каких случаях служащему ФНС может быть выплачена единовременная выплата на приобретение жилого помещения?\"\n"
    "\"Вправе ли инспектор ФНС принять ценный подарок от делегации дружественного государства?\"\n"
    "\"Что делать, если инспектору ФНС в социальной сети поступают угрозы от представителя проверяемого предприятия?\"\n\n"
    "Примеры USER:\n"
    "\"Какая из перечисленных ниже мер является профилактикой коррупции? Координация деятельности...\"\n"
    "\"В случае получения от руководителя неправомерного поручения как должен поступить госслужащий\"\n\n"
    "Запрос: {query}\n"
    "Ответ:"
)


# =========================================================
# Эвристика strong-EXACT и guard-fallback
# =========================================================
# Сильные реквизиты: наличие любого → EXACT без вызова LLM.
_STRONG_EXACT_REQUISITES = (
    "статья", "ст.", "фз", "указ", "постановление", "%", "процент",
)


def _guard_fallback(norm_query: str) -> Mode:
    """Guard-fallback: эвристика решения при недоступности LLM.

    EXACT — если в запросе есть цифры, реквизиты ИЛИ нормативные маркеры
    вопроса («в каких случаях», «вправе ли», «обязан ли» и т.п.).
    Иначе USER.
    """
    q_low = norm_query.lower()
    has_numbers = any(char.isdigit() for char in norm_query)
    has_exact_keywords = any(x in q_low for x in [
        "статья", "ст.", "фз", "указ", "постановление", "%", "процент",
        "в каких случаях", "в каком случае", "при каких условиях",
        "вправе ли", "обязан ли", "может ли", "допускается ли",
        "разрешено ли", "запрещено ли", "положено ли",
        "предусмотрена ли", "предусмотрено ли", "график", "круговая диаграмма", "столбчатая диаграмма",
    ])
    return "exact" if (has_numbers or has_exact_keywords) else "user"


def classify_mode(norm_query: str) -> Mode:
    """Классифицировать запрос в режим "exact" / "user".

    LLM-first: LLM — основной механизм; _guard_fallback — только fallback
    при недоступности LLM или непарсируемом ответе. Запросы с цифрами или
    сильными реквизитами идут в EXACT без вызова LLM (strong_exact).
    """
    if norm_query in _mode_cache:
        mode = _mode_cache[norm_query]
        weights = get_weights(mode)
        logger.info(
            "⚖️ Классификация: %s ➔ веса %s | источник=%s | LLM=%.2f сек",
            mode.upper(), weights, "cache", 0.0,
        )
        return mode

    mode: Mode = DEFAULT_MODE
    source: str = ""
    latency = 0.0

    # ШАГ 1: Очевидно EXACT — цифры или сильные реквизиты. LLM не тратим.
    q_low = norm_query.lower()
    has_numbers = any(char.isdigit() for char in norm_query)
    has_strong_requisites = any(x in q_low for x in _STRONG_EXACT_REQUISITES)

    if has_numbers or has_strong_requisites:
        mode = "exact"
        source = "strong_exact"
    else:
        # ШАГ 2: Всё остальное — решение отдаём LLM-классификатору.
        try:
            t0 = time.perf_counter()
            answer = _generate(
                _CLASSIFIER_PROMPT.format(query=norm_query),
                max_tokens=5,
                temperature=0.0,
            )
            latency = time.perf_counter() - t0
            answer_clean = answer.strip().lower().rstrip(".! ")

            if answer_clean == "exact":
                mode, source = "exact", "llm"
            elif answer_clean == "user":
                mode, source = "user", "llm"
            else:
                logger.warning(
                    "⚠️ Классификатор: неожиданный ответ %r → guard_fallback",
                    answer,
                )
                mode = _guard_fallback(norm_query)
                source = "guard_fallback"
        except Exception as e:
            logger.error(
                "❌ %s недоступна: %s → guard_fallback",
                CLASSIFIER_BACKEND, e,
            )
            mode = _guard_fallback(norm_query)
            source = "guard_fallback"

    weights = get_weights(mode)
    logger.info(
        "⚖️ Классификация: %s ➔ веса %s | источник=%s | LLM=%.2f сек",
        mode.upper(), weights, source, latency,
    )
    _mode_cache[norm_query] = mode
    return mode


def classify_query(norm_query: str) -> Tuple[float, float]:
    """Обратно-совместимая обёртка: возвращает веса для определённого режима."""
    return get_weights(classify_mode(norm_query))


def is_exact(norm_query: str) -> bool:
    """True если запрос EXACT (требует реранка и весов 0.5/0.5)."""
    return classify_mode(norm_query) == "exact"


def clear_cache() -> None:
    """Очистить кэш классификации (для тестов)."""
    _mode_cache.clear()


# =========================================================
# Кэш на уровне модуля
# =========================================================
_mode_cache: dict[str, Mode] = {}

__all__ = [
    "Mode", "WEIGHTS", "DEFAULT_MODE",
    "classify_mode", "classify_query", "is_exact",
    "get_weights", "clear_cache",
]
