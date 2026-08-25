"""Тесты полноты инструкций в QA-промптах.

Проверяет, что после фикса бага «LLM игнорирует часть контекста»
в системном промпте отсутствуют вредные правила и присутствуют
правила, требующие анализ ВСЕГО контекста для перечней.
Standalone — не требуют Qdrant, Ollama, RAG.
"""

import ast

ENGINE_FILES = {
    "app/rag/engine_rag.py": "multiline",
    "engines/engine_rag_bm25_e5.py": "concat",
    "engines/engine_rag_qdrant.py": "concat",
    "engines/engine_rag_e5.py": "concat",
    "engines/engine_rag_bm25_sber.py": "concat",
    "engines/engine_rag_light_rarank.py": "concat",
    "engines/engine_rag_gigachat.py": "multiline",
    "engines/engine_rag_qdrant_15052026.py": "multiline",
}


def _read_prompt_str(filepath: str) -> str:
    """Извлекает значение _QA_PROMPT_STR из файла через AST."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_QA_PROMPT_STR":
                    if isinstance(node.value, ast.Constant):
                        return str(node.value.value)
                    if isinstance(node.value, ast.JoinedStr):
                        parts = []
                        for v in node.value.values:
                            if isinstance(v, ast.Constant):
                                parts.append(str(v.value))
                            elif isinstance(v, ast.FormattedValue):
                                parts.append("{" + v.value.id + "}")
                        return "".join(parts)
                    return ast.unparse(node.value)
    return ""


def _read_file_content(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ==========================
#   ТЕСТЫ: НАЛИЧИЕ ПРАВИЛА ПОЛНОТЫ
# ==========================


def test_main_engine_has_completeness_rule():
    content = _read_file_content("app/rag/engine_rag.py")
    assert "ПЕРЕЧНИ (полнота ответа)" in content
    assert "проанализируй ВЕСЬ предоставленный КОНТЕКСТ" in content
    assert "Собери элементы из ВСЕХ релевантных норм" in content
    assert "ПРОВЕРКА ПОЛНОТЫ" in content
    assert "используй НАИБОЛЕЕ ПОЛНЫЙ из имеющихся перечней" in content


def test_main_engine_no_harmful_rule8():
    content = _read_file_content("app/rag/engine_rag.py")
    for phrase in ["отвечай ТОЛЬКО по тому пункту", "Условия из других пунктов ИГНОРИРУЙ"]:
        assert phrase not in content, f"Есть вредная фраза: {phrase!r}"


def test_main_engine_no_do_not_enumerate():
    content = _read_file_content("app/rag/engine_rag.py")
    assert "Не обобщай и не перечисляй всё подряд" not in content


def test_group_a_have_completeness_rule():
    for path in [
        "engines/engine_rag_bm25_e5.py", "engines/engine_rag_qdrant.py",
        "engines/engine_rag_e5.py", "engines/engine_rag_bm25_sber.py",
        "engines/engine_rag_light_rarank.py",
    ]:
        content = _read_file_content(path)
        assert "10. ПЕРЕЧНИ (полнота ответа)" in content, f"{path}: нет правила 10"
        assert "проанализируй ВЕСЬ предоставленный контекст" in content
        assert "Собери элементы из ВСЕХ релевантных норм" in content
        assert "11. ПРОВЕРКА ПОЛНОТЫ" in content, f"{path}: нет правила 11"
        assert "используй НАИБОЛЕЕ ПОЛНЫЙ из имеющихся перечней" in content


def test_group_a_no_do_not_generalize():
    for path in [
        "engines/engine_rag_bm25_e5.py", "engines/engine_rag_qdrant.py",
        "engines/engine_rag_e5.py", "engines/engine_rag_bm25_sber.py",
        "engines/engine_rag_light_rarank.py",
    ]:
        content = _read_file_content(path)
        assert "Не обобщай!" not in content, f"{path}: есть 'Не обобщай!'"


def test_group_a_rule6_uses_all_context():
    for path in [
        "engines/engine_rag_bm25_e5.py", "engines/engine_rag_qdrant.py",
        "engines/engine_rag_e5.py", "engines/engine_rag_bm25_sber.py",
        "engines/engine_rag_light_rarank.py",
    ]:
        content = _read_file_content(path)
        assert "используй все релевантные фрагменты контекста" in content


def test_group_c_not_touched():
    for path in ["engines/engine_rag_gigachat.py", "engines/engine_rag_qdrant_15052026.py"]:
        content = _read_file_content(path)
        assert "ПЕРЕЧНИ (полнота ответа)" not in content, f"{path}: не должно быть правила"


def test_prompt_assembly_contains_completeness():
    prompt_str = _read_prompt_str("app/rag/engine_rag.py")
    assert prompt_str
    assert "проанализируй ВЕСЬ предоставленный КОНТЕКСТ" in prompt_str
    assert "Собери элементы из ВСЕХ релевантных норм" in prompt_str
    assert "ПРОВЕРКА ПОЛНОТЫ" in prompt_str
    assert "используй НАИБОЛЕЕ ПОЛНЫЙ из имеющихся перечней" in prompt_str


def test_group_a_prompt_ast_extraction():
    for path in [
        "engines/engine_rag_bm25_e5.py", "engines/engine_rag_qdrant.py",
        "engines/engine_rag_e5.py", "engines/engine_rag_bm25_sber.py",
        "engines/engine_rag_light_rarank.py",
    ]:
        prompt_str = _read_prompt_str(path)
        assert prompt_str, f"Не извлечён _QA_PROMPT_STR из {path}"
        assert "10. ПЕРЕЧНИ (полнота ответа)" in prompt_str
        assert "проанализируй ВЕСЬ предоставленный контекст" in prompt_str
        assert "11. ПРОВЕРКА ПОЛНОТЫ" in prompt_str


def test_print_prompt_preview(capsys):
    """Выводит первые 500 символов промпта для визуальной проверки."""
    content = _read_file_content("app/rag/engine_rag.py")
    start = content.find("_QA_PROMPT_STR")
    if start >= 0:
        snippet = content[start:start + 2000]
        with capsys.disabled():
            print("\n=== _QA_PROMPT_STR (первые 2000 символов) ===")
            print(snippet[:2000])
            print("\n=== END ===")
    assert True
