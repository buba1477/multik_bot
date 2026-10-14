"""LLM-классификатор запросов: EXACT (0.5/0.5) vs USER (1.0/0.0).

Определяет, содержит ли запрос точные реквизиты / конкретную величину,
чтобы применить гибридные веса RRF с большим весом BM25 (0.5/0.5),
улучшая поиск по узким терминологическим запросам.

Единый источник правды по весам — словарь WEIGHTS. Режим (exact/user)
определяет LLM; маппинг режима в веса делает get_weights().
engine_rag.py читает режим через classify_mode() и маппит его в веса.

Backend: ollama | gigachat (env CLASSIFIER_BACKEND, по умолч. ollama).
Fallback: DEFAULT_MODE="user" при недоступности LLM (никогда не использует regex).
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
    with GigaChatSDK(credentials=GIGACHAT_KEY, verify_ssl_certs=False) as giga:
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
def classify_mode(norm_query: str) -> Mode:
    """Классифицировать запрос в режим "exact" / "user".

    Кэширует результат по norm_query (второй вызов не дёргает LLM).
    Fallback: DEFAULT_MODE при недоступности LLM или неожиданном ответе.
    """
    if norm_query in _mode_cache:
        return _mode_cache[norm_query]

    mode: Mode = DEFAULT_MODE
    latency = 0.0
    try:
        t0 = time.perf_counter()
        answer = _generate(_CLASSIFIER_PROMPT.format(query=norm_query), max_tokens=5, temperature=0.0)
        latency = time.perf_counter() - t0
        answer_clean = answer.strip().lower().rstrip(".! ")
        if answer_clean == "exact":
            mode = "exact"
        elif answer_clean == "user":
            mode = "user"
        else:
            logger.warning(
                "⚠️ Классификатор: неожиданный ответ %r → fallback USER", answer
            )
    except Exception as e:
        logger.error(
            "❌ %s недоступна: %s → fallback USER", CLASSIFIER_BACKEND, e
        )

    weights = get_weights(mode)
    label = "EXACT" if mode == "exact" else "USER"
    logger.info(
        "⚖️ Классификация [%s]: %s → (%.1f, %.1f) (latency=%.2fs)",
        CLASSIFIER_BACKEND,
        label,
        weights[0],
        weights[1],
        latency,
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
# Prompt
# =========================================================
_CLASSIFIER_PROMPT = (
    "Ты — маршрутизатор запросов к базе НПА ФНС России.\n\n"
    "У системы два поисковика:\n"
    "- DENSE ищет по смыслу. Хорош для перефразируемых вопросов: "
    "\"что такое НДФЛ\", \"как уволиться\", \"какие документы нужны\".\n"
    "- BM25 ищет по точным словам. Хорош, когда в запросе есть "
    "уникальный идентификатор: номер статьи, номер закона, "
    "название раздела или приложения.\n\n"
    "Твоя задача — выбрать, какой поисковик найдёт ответ лучше.\n"
    "Верни ровно ОДНО слово: EXACT (BM25) или USER (DENSE).\n\n"
    "Примеры:\n"
    "\"статья 346.14 НК РФ\" → EXACT\n"
    "\"146-ФЗ статья 100\" → EXACT\n"
    "\"146-ФЗ статья 101.4\" → EXACT\n"
    "\"146-ФЗ статья 137\" → EXACT\n"
    "\"Указ 112 приложение 1\" → EXACT\n"
    "\"Постановление 397 раздел VI\" → EXACT\n"
    "\"117-ФЗ статья 217\" → EXACT\n"
    "\"пункт 4 статьи 346.13\" → EXACT\n"
    "\"что такое НДФЛ\" → USER\n"
    "\"как уволиться с госслужбы\" → USER\n"
    "\"какие документы нужны для конкурса\" → USER\n"
    "\"какие сроки подачи декларации\" → USER\n"
    "\"кто обязан подавать сведения о доходах\" → USER\n"
    "\"какие взыскания применяются к госслужащим\" → USER\n\n"
    "Запрос: {query}\n"
    "Ответ:"
)

# =========================================================
# Кэш на уровне модуля
# =========================================================
_mode_cache: dict[str, Mode] = {}

__all__ = [
    "Mode", "WEIGHTS", "DEFAULT_MODE",
    "classify_mode", "classify_query", "is_exact",
    "get_weights", "clear_cache",
]
