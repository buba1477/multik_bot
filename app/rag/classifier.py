"""LLM-классификатор запросов: EXACT (0.5/0.5) vs USER (0.9/0.1).

Определяет, содержит ли запрос точные реквизиты / конкретную величину,
чтобы применить гибридные веса RRF с бóльшим весом BM25 (0.5/0.5),
улучшая поиск по узким терминологическим запросам.

Backend: ollama | gigachat (env CLASSIFIER_BACKEND, по умолч. ollama).
Fallback: (0.9, 0.1) при недоступности LLM (никогда не использует regex).
Кэш: _weights_cache — глобальный словарь модуля.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Tuple

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
# Веса для двух режимов
# =========================================================
USER_VECTOR_WEIGHT = 0.9
USER_BM25_WEIGHT = 0.1
EXACT_VECTOR_WEIGHT = 0.5
EXACT_BM25_WEIGHT = 0.5
# =========================================================
# Backend: Ollama
# =========================================================
def _ollama_generate(prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
    """Генерация через Ollama (urllib, без внешних зависимостей)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
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
    with urllib.request.urlopen(req, timeout=30) as resp:
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


# =========================================================
# =========================================================
# Публичный API
# =========================================================
def classify_query(norm_query: str) -> Tuple[float, float]:
    """Классифицировать запрос и вернуть (vector_weight, bm25_weight).

    Args:
        norm_query: Нормализованный текст запроса (без стоп-слов, с заменами).

    Returns:
        Кортеж (vector_weight, bm25_weight): EXACT → (0.5, 0.5), USER → (0.9, 0.1).
    """
    if not norm_query or not norm_query.strip():
        return (USER_VECTOR_WEIGHT, USER_BM25_WEIGHT)

    # Кэш
    if norm_query in _weights_cache:
        return _weights_cache[norm_query]

    prompt = _CLASSIFIER_PROMPT.format(query=norm_query)

    is_exact = False
    latency = 0.0
    try:
        t0 = time.perf_counter()
        answer = _generate(prompt, max_tokens=5, temperature=0.0)
        latency = time.perf_counter() - t0
        answer_clean = answer.strip().lower().rstrip(".! ")
        if answer_clean == "exact":
            is_exact = True
        elif answer_clean == "user":
            is_exact = False
        else:
            logger.warning(
                "⚠️ Классификатор: неожиданный ответ %r → fallback USER", answer
            )
    except Exception as e:
        logger.error(
            "❌ %s недоступна: %s → fallback USER", CLASSIFIER_BACKEND, e
        )

    weights = (
        (EXACT_VECTOR_WEIGHT, EXACT_BM25_WEIGHT)
        if is_exact
        else (USER_VECTOR_WEIGHT, USER_BM25_WEIGHT)
    )
    label = "EXACT" if is_exact else "USER"
    logger.info(
        "⚖️ Классификация [%s]: %s → (%.1f, %.1f) (latency=%.2fs)",
        CLASSIFIER_BACKEND,
        label,
        weights[0],
        weights[1],
        latency,
    )
    _weights_cache[norm_query] = weights
    return weights


def is_exact(norm_query: str) -> bool:
    """True если запрос EXACT (требует реранка и весов 0.5/0.5)."""
    weights = classify_query(norm_query)
    return weights == (EXACT_VECTOR_WEIGHT, EXACT_BM25_WEIGHT)
# Dispatching
# =========================================================
def _generate(prompt: str, max_tokens: int = 5, temperature: float = 0.0) -> str:
    """Выбор бэкенда по CLASSIFIER_BACKEND."""
    if CLASSIFIER_BACKEND == "gigachat":
        return _gigachat_generate(prompt, max_tokens, temperature)
    return _ollama_generate(prompt, max_tokens, temperature)


# =========================================================
# Prompt
# =========================================================
_CLASSIFIER_PROMPT = (
    "Ты — классификатор запросов к базе НПА ФНС России.\n\n"
    "Определи тип запроса и верни ровно ОДНО слово: EXACT или USER.\n\n"
    "EXACT — запрос содержит точные реквизиты или конкретную величину:\n"
    "- номер статьи/ФЗ/указа (117-ФЗ, статья 217, пункт 4, Указ 112)\n"
    "- точные цифры/термины (размер надбавки, 15 лет, 30 процентов, оклад, выслуга)\n"
    "- вопрос про конкретную ставку/размер/количество\n\n"
    "USER — запрос общего характера:\n"
    '- "что такое НДФЛ", "как уволиться", "какие документы"\n'
    "- общий вопрос без точных цифр и реквизитов\n\n"
    "Запрос: {query}\n"
    "Ответ:"
)

# =========================================================
# Кэш на уровне модуля
# =========================================================
_weights_cache: dict[str, Tuple[float, float]] = {}