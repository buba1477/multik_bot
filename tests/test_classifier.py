"""Тесты модуля app.rag.classifier — LLM-классификатор запросов.

Offline-тесты проверяют fallback (USER при недоступности LLM).
Mock-тесты проверяют логику классификации (EXACT/USER).
Integration-тесты требуют Ollama/GigaChat (запускать вручную).
"""
from unittest.mock import patch

from app.rag.classifier import is_exact, classify_query, _weights_cache


def clear_cache():
    """Очистить кэш перед каждым тестом."""
    _weights_cache.clear()


# =========================================================
# Offline-тесты (без LLM — тестируем fallback)
# =========================================================

def test_offline_fallback():
    """Без LLM все запросы → USER (fallback)."""
    clear_cache()
    assert is_exact("117-ФЗ статья 217") is False
    assert is_exact("Что такое НДФЛ?") is False


def test_offline_empty():
    """Пустой запрос → USER."""
    clear_cache()
    assert is_exact("") is False
    w = classify_query("")
    assert w == (0.9, 0.1)


def test_offline_cache():
    """Повторный запрос не вызывает LLM, но берётся из кэша."""
    clear_cache()
    w1 = classify_query("любой запрос")
    w2 = classify_query("любой запрос")
    assert w1 == w2


# =========================================================
# Mock-тесты (логика классификации)
# =========================================================

@patch("app.rag.classifier._generate", return_value="EXACT")
def test_mock_exact(mock_generate):
    """Mock: LLM вернул EXACT → is_exact=True."""
    clear_cache()
    assert is_exact("117-ФЗ статья 217") is True
    mock_generate.assert_called_once()


@patch("app.rag.classifier._generate", return_value="USER")
def test_mock_user(mock_generate):
    """Mock: LLM вернул USER → is_exact=False."""
    clear_cache()
    assert is_exact("Как уволиться?") is False
    mock_generate.assert_called_once()


@patch("app.rag.classifier._generate", return_value="EXACT")
def test_mock_classify_exact_weights(mock_generate):
    """Mock: EXACT → weights (0.5, 0.5)."""
    clear_cache()
    w = classify_query("надбавка за выслугу 15 лет")
    assert w == (0.5, 0.5)


@patch("app.rag.classifier._generate", return_value="USER")
def test_mock_classify_user_weights(mock_generate):
    """Mock: USER → weights (0.9, 0.1)."""
    clear_cache()
    w = classify_query("Как уволиться?")
    assert w == (0.9, 0.1)


@patch("app.rag.classifier._generate", return_value="SOMETHING_ELSE")
def test_mock_unexpected_answer(mock_generate):
    """Неожиданный ответ LLM → fallback USER."""
    clear_cache()
    w = classify_query("тест")
    assert w == (0.9, 0.1)


@patch("app.rag.classifier._generate", side_effect=Exception("Ollama timeout"))
def test_mock_exception(mock_generate):
    """Ошибка LLM → fallback USER."""
    clear_cache()
    w = classify_query("тест")
    assert w == (0.9, 0.1)


@patch("app.rag.classifier._generate", return_value="EXACT")
def test_mock_cache(mock_generate):
    """Повторный запрос из кэша → _generate вызывается единожды."""
    clear_cache()
    w1 = classify_query("повтор")
    w2 = classify_query("повтор")
    assert w1 == w2
    mock_generate.assert_called_once()


# =========================================================
# Integration-тесты (требуют Ollama/GigaChat)
# Запуск: ollama: pytest tests/test_classifier.py -v -k "integration"
#         gigachat: CLASSIFIER_BACKEND=gigachat pytest ... -k "integration"
# =========================================================

def test_integration_exact():
    """Integration: ФЗ-запрос. Требует LLM."""
    import os
    import pytest
    if not os.environ.get("CLASSIFIER_BACKEND"):
        pytest.skip("SKIP: CLASSIFIER_BACKEND не задан")
    clear_cache()
    assert is_exact("117-ФЗ статья 217") is True


def test_integration_target():
    """Integration: целевой запрос про надбавку. Требует LLM."""
    import os
    import pytest
    if not os.environ.get("CLASSIFIER_BACKEND"):
        pytest.skip("SKIP: CLASSIFIER_BACKEND не задан")
    clear_cache()
    q = (
        "Размер ежемесячной надбавки к должностному окладу за выслугу лет "
        "на государственной гражданской службе при стаже свыше 15 лет составляет?"
    )
    assert is_exact(q) is True


def test_integration_user():
    """Integration: общий вопрос. Требует LLM."""
    import os
    import pytest
    if not os.environ.get("CLASSIFIER_BACKEND"):
        pytest.skip("SKIP: CLASSIFIER_BACKEND не задан")
    clear_cache()
    assert is_exact("Что такое НДФЛ?") is False
    assert is_exact("Как уволиться с госслужбы?") is False