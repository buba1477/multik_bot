"""Тесты модуля app.rag.classifier — LLM-классификатор запросов.

Offline-тесты проверяют strong_exact и guard-fallback (когда LLM недоступна).
Mock-тесты проверяют LLM-first логику классификации (EXACT/USER).
Integration-тесты требуют Ollama/GigaChat (запускать вручную).
"""
from unittest.mock import patch

import pytest

from app.rag.classifier import (
    _guard_fallback,
    classify_mode,
    classify_query,
    clear_cache,
    is_exact,
)


@pytest.fixture(autouse=True)
def _clear():
    """Независимость тестов: чистый кэш до и после каждого теста."""
    clear_cache()
    yield
    clear_cache()


# =========================================================
# LLM-first классификация (приемочные сценарии из ТЗ)
# =========================================================

@pytest.mark.parametrize("query,expected", [
    (
        "В каких случаях служащему ФНС может быть выплачена единовременная "
        "выплата на приобретение жилого помещения?",
        "exact",
    ),
    (
        "Вправе ли инспектор ФНС принять ценный подарок от делегации "
        "дружественного государства?",
        "exact",
    ),
    (
        "Что делать, если инспектору ФНС в социальной сети поступают угрозы?",
        "exact",
    ),
    ("статья 122 НК РФ", "exact"),
    ("Расскажи про увольнение за утрату доверия", "user"),
])
def test_classify_mode(query: str, expected: str):
    """Нормативные вопросы без реквизитов классифицирует LLM.

    «статья 122 НК РФ» уходит в strong_exact (без вызова LLM),
    остальные запросы решает мок-LLM (_generate).
    """
    with patch("app.rag.classifier._generate", return_value=expected.upper()):
        assert classify_mode(query) == expected


# =========================================================
# guard-fallback: раздельные сценарии exact/user
# =========================================================

def test_guard_fallback_exact_cases():
    """«В каких случаях…» есть в guard-маркерах → fallback вернёт exact."""
    q = "В каких случаях служащему ФНС положена выплата?"
    assert _guard_fallback(q) == "exact"


def test_guard_fallback_user_for_what_to_do():
    """«Что делать…» НЕ входит в guard-маркеры → fallback вернёт user."""
    q = "Что делать, если инспектору ФНС в социальной сети поступают угрозы?"
    assert _guard_fallback(q) == "user"


@patch("app.rag.classifier._generate", return_value="EXACT")
def test_llm_exact_for_what_to_do(mock_generate):
    """«Что делать…» → exact достигается ТОЛЬКО через LLM (не через fallback)."""
    q = "Что делать, если инспектору ФНС в социальной сети поступают угрозы?"
    assert classify_mode(q) == "exact"
    mock_generate.assert_called_once()


# =========================================================
# strong_exact и кэш экономят вызовы LLM
# =========================================================

@patch(
    "app.rag.classifier._generate",
    side_effect=AssertionError("LLM не должен вызываться для strong_exact"),
)
def test_strong_exact_skips_llm(mock_generate):
    """«статья 122 НК РФ» → exact через strong_exact, _generate вызывается 0 раз."""
    assert classify_mode("статья 122 НК РФ") == "exact"
    assert mock_generate.call_count == 0


@patch("app.rag.classifier._generate", return_value="USER")
def test_cache_skips_llm(mock_generate):
    """Повторный вызов с тем же запросом берётся из кэша без повторного вызова LLM."""
    q = "Расскажи про увольнение за утрату доверия"
    assert classify_mode(q) == "user"   # 1-й вызов → LLM (мок вернул USER)
    assert mock_generate.call_count == 1
    assert classify_mode(q) == "user"   # 2-й вызов → кэш, LLM не вызывается
    assert mock_generate.call_count == 1


# =========================================================
# Offline-тесты (без LLM — strong_exact и guard fallback)
# =========================================================

def test_offline_fallback():
    """Без LLM: реквизитный запрос → EXACT (strong_exact), смысловой → USER (guard fallback)."""
    assert is_exact("117-ФЗ статья 217") is True
    assert is_exact("Что такое НДФЛ?") is False


def test_offline_empty():
    """Пустой запрос → USER."""
    assert is_exact("") is False
    w = classify_query("")
    assert w == (1.0, 0.0)


def test_offline_cache():
    """Повторный запрос не вызывает LLM, но берётся из кэша."""
    w1 = classify_query("любой запрос")
    w2 = classify_query("любой запрос")
    assert w1 == w2


# =========================================================
# Mock-тесты (LLM-first логика)
# =========================================================

_EXACT_LLM_QUERY = (
    "Какие ограничения установлены для гражданских служащих "
    "при получении подарков?"
)


@patch("app.rag.classifier._generate", return_value="EXACT")
def test_mock_exact(mock_generate):
    """LLM вернул EXACT → is_exact=True.

    Запрос не содержит сильных реквизитов и guard-маркеров — точность
    берётся именно из LLM (_generate вызван ровно один раз).
    """
    assert is_exact(_EXACT_LLM_QUERY) is True
    mock_generate.assert_called_once()


@patch("app.rag.classifier._generate", return_value="USER")
def test_mock_user(mock_generate):
    """Mock: LLM вернул USER → is_exact=False."""
    assert is_exact("Как уволиться?") is False
    mock_generate.assert_called_once()


@patch("app.rag.classifier._generate", return_value="EXACT")
def test_mock_classify_exact_weights(mock_generate):
    """Mock: EXACT → weights (0.5, 0.5)."""
    assert classify_query(_EXACT_LLM_QUERY) == (0.5, 0.5)
    mock_generate.assert_called_once()


@patch("app.rag.classifier._generate", return_value="USER")
def test_mock_classify_user_weights(mock_generate):
    """Mock: USER → weights (1.0, 0.0)."""
    assert classify_query("Как уволиться?") == (1.0, 0.0)


@patch("app.rag.classifier._generate", return_value="SOMETHING_ELSE")
def test_mock_unexpected_answer(mock_generate):
    """Неожиданный ответ LLM → guard_fallback (эвристика)."""
    assert classify_query("тест") == (1.0, 0.0)


@patch("app.rag.classifier._generate", side_effect=Exception("Ollama timeout"))
def test_mock_exception(mock_generate):
    """Ошибка LLM → guard_fallback (эвристика)."""
    assert classify_query("тест") == (1.0, 0.0)


# =========================================================
# Integration-тесты (требуют Ollama/GigaChat)
# =========================================================

def test_integration_exact():
    """Integration: ФЗ-запрос. Требует LLM."""
    import os
    import pytest
    if not os.environ.get("CLASSIFIER_BACKEND"):
        pytest.skip("SKIP: CLASSIFIER_BACKEND не задан")
    assert is_exact("117-ФЗ статья 217") is True


def test_integration_target():
    """Integration: целевой запрос про надбавку. Требует LLM."""
    import os
    import pytest
    if not os.environ.get("CLASSIFIER_BACKEND"):
        pytest.skip("SKIP: CLASSIFIER_BACKEND не задан")
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
    assert is_exact("Что такое НДФЛ?") is False
    assert is_exact("Как уволиться с госслужбы?") is False
