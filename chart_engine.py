import json
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from app_logger import logger

# =========================================================
# CONTRACT SCHEMA (Жесткая структура для Apache ECharts)
# =========================================================
class EChartsConfig(BaseModel):
    title: str = Field(description="Заголовок графика, отражающий суть извлеченных данных.")
    chart_type: str = Field(description="Тип графика: строго 'bar' (столбчатый), 'line' (линейный) или 'pie' (круговая диаграмма).")
    x_axis: Optional[List[str]] = Field(default=None, description="Массив названий категорий для оси X (для 'pie' это будут названия секторов).")
    series_name: str = Field(description="Название серии данных, например: 'Календарные дни'.")
    series_data: List[float] = Field(description="Массив числовых значений для оси Y. Только плоский массив чисел, БЕЗ вложенных списков.")
    series_name_2: Optional[str] = Field(default=None, description="Название второй серии данных (для двух рядов на одном графике).")
    series_data_2: Optional[List[float]] = Field(default=None, description="Массив числовых значений для второй серии данных (для двух рядов на одном графике).")
    x_axis_2: Optional[List[str]] = Field(default=None, description="Массив названий категорий для второй оси X (если отличается).")
    y_axis_label: Optional[str] = Field(default=None, description="Подпись оси Y с единицей измерения, например: 'Доля, %', 'Сумма, тыс. руб.', 'Количество, чел.'.")
    unit: Optional[str] = Field(default=None, description="Единица измерения значений на графике, например: '%', 'руб.', 'чел.', 'млн руб'.")

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

            ВАЖНЕЙШЕЕ ПРАВИЛО: В контексте может быть НЕСКОЛЬКО независимых таблиц с РАЗНЫМИ единицами измерения (например, млрд руб, млн чел, %). 
            Ты должен выбрать ТОЛЬКО ОДНУ таблицу, наиболее соответствующую запросу пользователя.
            ЗАПРЕЩЕНО смешивать данные из разных таблиц в одном графике.
            ЗАПРЕЩЕНО класть значения с разными единицами измерения в один series_data.

            Шаблон (простой график, один ряд данных):
            {
            "title": "Заголовок на русском",
            "chart_type": "bar, line или pie",
            "x_axis": ["Категория 1", "Категория 2"],
            "series_name": "Название серии",
            "series_data": [число_1, число_2],
            "y_axis_label": "Подпись оси Y с единицей, например: 'Доля, %', 'Сумма, руб.', 'Количество, чел.' (если применимо)",
            "unit": "Единица измерения: '%', 'руб.', 'чел.', 'млн руб.' и т.д. (только если есть в данных)"
            }

            Шаблон (сравнение, два ряда данных на одном графике):
            {
            "title": "Заголовок на русском",
            "chart_type": "bar или line",
            "x_axis": ["2023", "2024"],
            "series_name": "Первый показатель",
            "series_data": [100, 200],
            "series_name_2": "Второй показатель",
            "series_data_2": [300, 400],
            "y_axis_label": "Подпись оси Y с единицей",
            "unit": "Единица измерения"
            }

            СТРОГИЕ ПРАВИЛА:
            1. series_data и series_data_2 — это ТОЛЬКО плоский массив чисел, например [7150, 8120].
               ЗАПРЕЩЕНО использовать вложенные списки: [[2023, 7150], [2024, 8120]] — это ОШИБКА.
            2. Если в данных есть годы (например, 2023, 2024), вынеси их в x_axis, а значения — в series_data.
            3. Если нужно сравнить два разных показателя за одни и те же годы, используй series_name_2 и series_data_2.
            4. Числа могут быть как целыми, так и дробными (например, 8120.4).
            5. Размерность x_axis и series_data / series_data_2 должна строго совпадать.
            6. ВАЖНО: Если в данных явно указаны единицы измерения (%, руб., млн руб., чел., тыс. руб. и т.д.), ОБЯЗАТЕЛЬНО заполни поля "y_axis_label" (например "Доля, %") и "unit" (например "%"). Если единиц нет — оставь null.
            7. ВАЖНО: Для ЛЮБОГО типа графика (bar, line, pie) извлекай ВСЕ строки из выбранной таблицы, а не только итоговую или первую строку. Например, если в таблице 8 показателей (Консолидированный бюджет РФ, Федеральный бюджет, НДПИ, НДС и т.д.) — все 8 должны попасть в x_axis и series_data. НЕ обрезай данные до 1-2 категорий.
            8. ГЛАВНОЕ ПРАВИЛО: Выводи данные ТОЧНО КАК В ТЕКСТЕ. ЗАПРЕЩЕНО домножать, переводить, изменять единицы измерения или числа. Если в тексте написано "3,5 трлн руб." — в series_data пиши 3.5, а в unit пиши "трлн руб.". Если в тексте "7 102,4 млрд руб." — в series_data пиши 7102.4, а в unit пиши "млрд руб.". НЕ МЕНЯЙ данные, выводи как есть.
            9. ВАЖНО: Не смешивай данные с разными единицами измерения в одном графике. Каждый график — одна единица измерения.
            10. ВАЖНО: Никогда не помещай годы внутрь series_data!
            11. КРИТИЧЕСКОЕ ПРАВИЛО: Если в предоставленном тексте Базы знаний вообще НЕТ числовых показателей, таблиц или годов по запросу пользователя, и извлечь данные невозможно — ЗАПРЕЩЕНО выводить пустой JSON или null. Выведи JSON, где в поле "title" напиши "Данные отсутствуют", а в поле "series_data" поставь массив из одного нуля [0], в "x_axis" — ["Нет данных"]."""
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
        
        if is_chart and chart_type:
            # Принудительно проставляем тип графика в конец промпта
            sys_prompt = sys_prompt + f'\nОБЯЗАТЕЛЬНО: chart_type должен быть "{chart_type}". Строго следуй этому типу!'
        
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

    def _expand_short_numbers(self, obj: dict) -> dict:
        """Пост-обработка чисел из series_data.
        
        УБИРАЕМ умножение на миллионы/миллиарды/триллионы, так как оно ломает масштаб осей.
        Вместо этого гарантируем, что правильная единица измерения запишется в y_axis_label,
        а числа останутся компактными (например, 3.5 или 7102.4).
        """
        unit = (obj.get("unit") or "").strip()
        
        if unit:
            # Если модель заполнила unit (например, "трлн руб."), но забыла сделать красивую подпись для оси Y
            if not obj.get("y_axis_label"):
                obj["y_axis_label"] = f"Значение, {unit}"
        
        # Мы просто возвращаем объект с оригинальными компактными числами от LLM.
        # Если модель выдала 3.5, на графике будет 3.5, а на оси Y будет написано "Значение, трлн руб."
        return obj

    def _sanitize_json(self, content: str) -> str:
        """Санитизирует JSON перед Pydantic: null → 0, float → int через round(), распаковывает вложенные списки"""
        try:
            obj = json.loads(content)
            
            def flatten_series(values):
                """Преобразует [2023, 7150] в 7150, оставляет числа как есть"""
                if not isinstance(values, list):
                    return []
                result = []
                for v in values:
                    if v is None:
                        result.append(0)
                    elif isinstance(v, list):
                        if len(v) >= 2:
                            val = v[1] if v[1] is not None else 0
                            while isinstance(val, list):
                                val = val[-1] if val else 0
                            result.append(float(val) if isinstance(val, (int, float)) else 0)
                        else:
                            result.append(0)
                    elif isinstance(v, (int, float)):
                        result.append(float(v))
                    else:
                        try:
                            result.append(float(v))
                        except (ValueError, TypeError):
                            result.append(0)
                return result
            
            # 1. Обрабатываем series_data
            if "series_data" in obj and isinstance(obj["series_data"], list):
                obj["series_data"] = flatten_series(obj["series_data"])
            
            # 2. Обрабатываем series_data_2, если есть
            if "series_data_2" in obj and isinstance(obj["series_data_2"], list):
                obj["series_data_2"] = flatten_series(obj["series_data_2"])
            
            # =========================================================
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: СИНХРОНИЗАЦИЯ ОСЕЙ И ЧИСЕЛ
            # =========================================================
            if "series_data" in obj and isinstance(obj["series_data"], list):
                data_len = len(obj["series_data"])
                
                # Если x_axis пустой, null или его нет — создаем дефолтные имена
                if "x_axis" not in obj or not isinstance(obj["x_axis"], list) or not obj["x_axis"]:
                    obj["x_axis"] = [f"Показатель {i+1}" for i in range(data_len)]
                
                # Если модель сгенерировала мало категорий — дописываем недостающие
                elif len(obj["x_axis"]) < data_len:
                    current_len = len(obj["x_axis"])
                    for i in range(current_len, data_len):
                        obj["x_axis"].append(f"Показатель {i+1}")
                        
                # Если категорий больше чем чисел — обрезаем лишние
                elif len(obj["x_axis"]) > data_len:
                    obj["x_axis"] = obj["x_axis"][:data_len]

            # Нормализуем x_axis_2
            if "x_axis_2" not in obj or obj["x_axis_2"] is None:
                obj["x_axis_2"] = []

            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            return content


    def _validate_data_integrity(self, obj: dict) -> tuple[bool, str]:
        """
        Проверяет целостность данных после парсинга.
        Возвращает (True, "") если всё ок, или (False, "причина ошибки").
        Проверяет:
        - Нет ли в данных значений с разными порядками (отличаются > 100x)
        - Не смешаны ли разные единицы измерения
        """
        data_arrays = []
        if "series_data" in obj and isinstance(obj["series_data"], list):
            data_arrays.append(obj["series_data"])
        if "series_data_2" in obj and isinstance(obj["series_data_2"], list):
            data_arrays.append(obj["series_data_2"])
        
        if not data_arrays:
            return False, "Нет данных для построения графика"
        
        for arr in data_arrays:
            if not arr:
                continue
            # Фильтруем только положительные числа для анализа порядков
            pos_values = [v for v in arr if isinstance(v, (int, float)) and v > 0]
            if len(pos_values) < 2:
                continue
            
            min_val = min(pos_values)
            max_val = max(pos_values)
            
            # Если разница в порядках > 100 — это разные таблицы
            if min_val > 0 and max_val / min_val > 100:
                return False, (
                    f"Данные имеют слишком большой разброс значений "
                    f"(мин={min_val}, макс={max_val}, соотношение={max_val/min_val:.0f}x). "
                    f"График с такими данными будет некорректным."
                )
        
        return True, ""
    
    def validate_and_parse(self, raw_content: str) -> dict:
        """Валидирует JSON через контракт Pydantic и страхует фронтенд от краша"""
        content = raw_content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        
        # Перед валидацией проверяем, а не пустой ли JSON прислала модель из-за отсутствия данных
        try:
            test_obj = json.loads(content)
            # Если ключевые поля пустые — значит данных в тексте не было
            if not test_obj.get("series_data") or test_obj.get("series_data") == []:
                return {
                    "type": "chart_error",
                    "message": "⚠️ В базе знаний нет числовых данных для построения этого графика. Пожалуйста, уточните запрос или проверьте контекст.",
                    "raw": content
                }
        except Exception:
            pass

        content = self._sanitize_json(content)
            
        try:
            validated_data = EChartsConfig.model_validate_json(content)
            validated_dict = validated_data.model_dump()
            
            return {
                "type": "chart",
                "data": validated_dict
            }
        except Exception as e:
            logger.warning(f"⚠️ [PYDANTIC VALIDATION ERROR]: {e}")
            return {
                "type": "chart_error",
                "message": "⚠️ Не удалось извлечь структуру данных для графика. Попробуйте переформулировать запрос.",
                "raw": content[:200]
            }
