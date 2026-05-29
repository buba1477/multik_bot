import json
import re
import requests
from pydantic import BaseModel, Field
from typing import List

# =========================================================
# CONTRACT SCHEMA (Жесткая структура для Apache ECharts)
# =========================================================
class EChartsConfig(BaseModel):
    title: str = Field(description="Заголовок графика, отражающий суть извлеченных данных.")
    chart_type: str = Field(description="Тип графика: строго 'bar' (столбчатый), 'line' (линейный) или 'pie' (круговая диаграмма).")
    x_axis: List[str] = Field(description="Массив названий категорий для оси X (для 'pie' это будут названия секторов).")
    series_name: str = Field(description="Название серии данных, например: 'Календарные дни'.")
    series_data: List[int] = Field(description="Массив числовых значений для оси Y (строго целые числа).")

# =========================================================
# CORE MODULE
# =========================================================
class DynamicChartEngine:
    def __init__(self, ollama_url: str = "http://ollama_container:11434"):
        self.ollama_url = f"{ollama_url}/api/chat"
        # Генерируем JSON-схему на основе Pydantic один раз при старте
        # self.chart_schema = EChartsConfig.model_json_schema()

    def is_chart_request(self, query: str) -> bool:
        """Интеллектуальный классификатор намерений (Intent Classifier)"""
        triggers = {"график", "диаграмма", "нарисуй", "схема", "визуализируй", "гистограмма", "круговая", "линейный", "pie", "line", "bar"}
        # Очищаем и разбиваем запрос на токены для точного поиска пересечений
        query_words = set(re.findall(r'[а-яёa-z0-9]+', query.lower()))
        return bool(query_words & triggers)
    
    def get_system_prompt(self, is_chart: bool) -> str:
        if is_chart:
            return """Ты — аналитик ФНС. Извлеки показатели из текста и верни СТРОГО чистый JSON-объект по шаблону, без markdown-оберток (```json) и пояснений.

                Шаблон:
                {
                "title": "Заголовок на русском",
                "chart_type": "bar или line или pie",
                "x_axis": ["Категория 1", "Категория 2"],
                "series_name": "Единица измерения",
                "series_data": [число_1, число_2]
                }

                Правила выбора 'chart_type':
                - 'pie' (Круговая): для долей, частей, структуры, процентов или если явно просят "пирог/круговой".
                - 'line' (Линейный): для динамики, трендов, изменения по годам/месяцам или если просят "линию".
                - 'bar' (Столбчатый): дефолт для сравнения независимых категорий или если просят просто "график".

                Важно: числа в 'series_data' строго целые (int). Размерность 'x_axis' и 'series_data' должна строго совпадать."""
        else:
            return "Ты — официальный ИИ-ассистент ФНС. Отвечай строго по фактам из Базы знаний. Пиши лаконично, используй списки."



    def detect_chart_type(self, query: str) -> str:
        """Детектит тип графика напрямую из запроса ДО отправки LLM"""
        q = query.lower()
        # Сначала точные совпадения
        if any(w in q for w in ["круговая", "пирог", "pie", "секторная", "доли", "проценты", "распределение"]):
            return "pie"
        if any(w in q for w in ["линейный", "тренд", "линия", "line", "динамика", "изменение", "помесячно"]):
            return "line"
        # По умолчанию bar
        return "bar"

    def process_llm_payload(self, query: str, rag_context: str, model_name: str) -> dict:
        """Формирует тело запроса и параметры для контейнера Ollama"""
        is_chart = self.is_chart_request(query)
        chart_type = self.detect_chart_type(query) if is_chart else None
        sys_prompt = self.get_system_prompt(is_chart)
        user_content = f"БАЗА ЗНАНИЙ ФНС:\n{rag_context}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}"
        
        if is_chart:
            # Передаём в промпт уже определённый тип графика вместо "выбери_правильный_тип"
            sys_prompt = sys_prompt.replace("выбери_правильный_тип", chart_type)
            # Раскомментируем инструкцию по типу
            sys_prompt = sys_prompt.replace('"выбери_правильный_тип"', f'"{chart_type}"')
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "options": {
                "temperature": 0.0,  # Зажимаем в ноль для тотального детерминизма
                "top_p": 0.1,
                "num_ctx": 6144      # Удерживаем кэш в пределах 6 ГБ VRAM
            },
            "stream": False
        }
        
        if is_chart:
            payload["format"] = "json"  # Включаем нативный JSON-mode Ollama
            
        return {"is_chart": is_chart, "payload": payload, "chart_type": chart_type}

    def _sanitize_json(self, content: str) -> str:
        """Санитизирует JSON перед Pydantic: null → 0, float → int через round()"""
        try:
            obj = json.loads(content)
            # Проходим по series_data и заменяем null/None на 0, float округляем
            if "series_data" in obj and isinstance(obj["series_data"], list):
                cleaned = []
                for v in obj["series_data"]:
                    if v is None:
                        cleaned.append(0)
                    elif isinstance(v, float):
                        cleaned.append(int(round(v)))
                    else:
                        cleaned.append(v)
                obj["series_data"] = cleaned
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            return content

    def validate_and_parse(self, raw_content: str) -> dict:
        """Валидирует JSON через контракт Pydantic и страхует фронтенд от краша"""
        content = raw_content.strip()
        # Срезаем случайную markdown-разметку, если модель проглючило
        if content.startswith("```"):
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        
        # Санитизация: заменяем null в series_data на 0
        content = self._sanitize_json(content)
            
        try:
            validated_data = EChartsConfig.model_validate_json(content)
            return {
                "type": "chart",
                "data": validated_data.model_dump()
            }
        except Exception as e:
            print(f"⚠️ [PYDANTIC VALIDATION ERROR]: Модель выдала кривую структуру: {e}")
            return {
                "type": "chart_error",
                "message": f"Ошибка генерации структуры графика: {e}",
                "raw": content[:200]
            }
