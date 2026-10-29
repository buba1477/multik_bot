import json
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from ..logger import logger

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

            ПРИМЕР (реальные данные из Базы знаний ФНС):
            Контекст: "Гражданским служащим предоставляется ежегодный основной оплачиваемый отпуск продолжительностью 30 календарных дней... дополнительный отпуск за выслугу лет: от 1 до 5 лет — 1 календарный день, от 5 до 10 лет — 5 календарных дней, от 10 до 15 лет — 7 календарных дней, 15 лет и более — 10 календарных дней."

            Правильный ответ для этого контекста:
            {
            "title": "Продолжительность дополнительного отпуска за выслугу лет",
            "chart_type": "bar",
            "x_axis": ["1-5 лет", "5-10 лет", "10-15 лет", "15+ лет"],
            "series_name": "Дополнительный отпуск, кал. дней",
            "series_data": [1, 5, 7, 10],
            "y_axis_label": "Продолжительность, кал. дней",
            "unit": "кал. дней"
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
        """
        ЖЕСТКИЙ ПОСТ-ПРОЦЕССИНГ ДЛЯ ИСПРАВЛЕНИЯ БАГА С ТРИЛЛИОНАМИ.
        Если модель выдала гигантские миллиарды (>= 1000) для круговой диаграммы,
        мы переводим их в человеческие триллионы.
        """
        unit = (obj.get("unit") or "").strip().lower()
        
        # Перехватываем косяк: если числа в миллиардах и они больше или равны 1000 (например, 3500 млрд)
        if "млрд" in unit and "series_data" in obj and isinstance(obj["series_data"], list):
            if any(v >= 1000 for v in obj["series_data"] if isinstance(v, (int, float))):
                
                # Переводим числа в триллионы (делим на 1000)
                obj["series_data"] = [round(v / 1000, 2) for v in obj["series_data"]]
                
                # Если есть второй ряд данных — его тоже переводим
                if "series_data_2" in obj and isinstance(obj["series_data_2"], list):
                    obj["series_data_2"] = [round(v / 1000, 2) for v in obj["series_data_2"]]
                
                # Корректируем единицу измерения на триллионы
                unit = unit.replace("млрд", "трлн")
                obj["unit"] = unit
                if obj.get("y_axis_label"):
                    obj["y_axis_label"] = obj["y_axis_label"].lower().replace("млрд", "трлн")
        
        # Гарантируем заполнение y_axis_label, если unit есть, а подписи нет
        if unit and not obj.get("y_axis_label"):
            obj["y_axis_label"] = f"Значение, {obj.get('unit')}"
            
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

    def _detect_chart_type(self, user_query: str) -> str:
        """Определяет тип графика из запроса пользователя."""
        q = (user_query or "").lower()
        if any(w in q for w in ["круг", "pie", "доля", "сектор"]):
            return "pie"
        if any(w in q for w in ["линия", "line", "динамик", "тренд"]):
            return "line"
        return "bar"

    def _make_title(self, key: str, user_query: str = "") -> str:
        """Формирует заголовок графика: из запроса или из ключа JSON."""
        if key == "data" and user_query:
            t = user_query
            for prefix in ["Нарисуй", "Сделай", "Построй", "Покажи"]:
                t = t.replace(prefix, "")
            t = t.strip().strip(".,!?:;")
            if t:
                # Обрезаем только при реальной необходимости и по границе слова,
                # иначе заголовок рвётся на полуслове ("...на гражданск").
                if len(t) > 150:
                    t = t[:150].rsplit(" ", 1)[0].rstrip(" ,;:-")
                return t.capitalize()
        title = str(key).replace("_", " ").strip().capitalize() or "Данные"
        return title

    def _extract_from_raw_json(self, raw_json: str, user_query: str = "") -> Optional[dict]:
        """
        Универсальный парсер: извлекает данные из произвольного JSON модели.
        Поддерживает два формата:
          A) вложенные пары "категория": "значение"
             {"график_отпуска": {"1-5 лет": "1 день", "5-10 лет": "5 дней"}}
          B) массив записей (таблица)
             {"data": [{"Стаж": "до 5 лет", "Отпуск": "1 день"}, ...]}
        Возвращает dict для EChartsConfig или None, если данные не извлечены.
        """
        chart_type = self._detect_chart_type(user_query)

        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        # ---- ФОРМАТ A: вложенный объект с парами "категория": "значение" ----
        for key, value in obj.items():
            if isinstance(value, dict) and len(value) >= 2:
                x_axis = []
                series_data = []
                for cat, val in value.items():
                    x_axis.append(str(cat))
                    # Извлекаем первое число из строки (например, "1 календарный день" → 1)
                    match = re.search(r'\d+', str(val))
                    if match:
                        series_data.append(float(match.group()))
                    else:
                        series_data.append(0.0)
                if len(series_data) >= 2:
                    title = self._make_title(key, user_query)
                    # Защита от мусорных ключей (labels/values и т.п.)
                    garbage_keys = {"labels", "values", "key", "value", "data", "item", "category"}
                    if all(str(x).strip().lower() in garbage_keys for x in x_axis):
                        logger.warning(f"⚠️ [CHART] Мусорные данные от модели (x_axis={x_axis}), отмена графика.")
                        return None
                    return {
                        "title": title,
                        "chart_type": chart_type,
                        "x_axis": x_axis,
                        "series_name": "Значение",
                        "series_data": series_data,
                        "unit": None,
                    }

        # ---- ФОРМАТ B: массив записей (таблица) ----
        for key, value in obj.items():
            if (
                isinstance(value, list)
                and len(value) >= 2
                and all(isinstance(item, dict) for item in value)
            ):
                records = value
                col_names = list(records[0].keys())
                if len(col_names) < 2:
                    continue

                # Определяем колонки: категориальная и числовая
                col_values = {
                    col: [str(r.get(col, "")) for r in records] for col in col_names
                }
                cat_col = None
                val_col = None
                for col in col_names:
                    vals = col_values[col]
                    has_units = any(
                        re.search(r'(дн|дней|кал|руб|%|шт|тыс)', v, re.I) for v in vals
                    )
                    pure_numbers = [
                        v for v in vals if re.fullmatch(r'\d+(\.\d+)?', v.strip())
                    ]
                    has_ranges = any(
                        re.search(r'(от|до|—|–|\d+\s*[-–]\s*\d+)', v, re.I)
                        for v in vals
                    )
                    if has_units or (pure_numbers and not has_ranges):
                        val_col = col
                    elif not pure_numbers or has_ranges:
                        cat_col = col

                if cat_col is None:
                    cat_col = col_names[0]
                if val_col is None:
                    val_col = col_names[1] if len(col_names) > 1 else col_names[0]

                x_axis = []
                series_data = []
                for i, rec in enumerate(records):
                    x_axis.append(str(rec.get(cat_col, f"Категория {i+1}")))
                    match = re.search(r'\d+', str(rec.get(val_col, "")))
                    series_data.append(float(match.group()) if match else 0.0)

                if len(series_data) >= 2:
                    title = self._make_title(key, user_query)
                    # Защита от мусорных ключей (labels/values и т.п.)
                    garbage_keys = {"labels", "values", "key", "value", "data", "item", "category"}
                    if all(str(x).strip().lower() in garbage_keys for x in x_axis):
                        logger.warning(f"⚠️ [CHART] Мусорные данные от модели (x_axis={x_axis}), отмена графика.")
                        return None
                    return {
                        "title": title,
                        "chart_type": chart_type,
                        "x_axis": x_axis,
                        "series_name": str(val_col),
                        "series_data": series_data,
                        "unit": None,
                    }

        return None

    def _detect_unit_from_context(self, text: str) -> Optional[str]:
        """Определяет единицу измерения по тексту контекста/метаданных."""
        if not text:
            return None
        if re.search(r"%|процент", text, re.I):
            return "%"
        if re.search(r"руб", text, re.I):
            return "руб."
        if re.search(r"дн|дней|календарн", text, re.I):
            return "дн."
        return None

    def _extract_table_from_context(self, rag_context: str, user_query: str = "") -> Optional[dict]:
        """
        Извлекает табличные данные из текста RAG-контекста (регулярками).
        Ищет паттерны вида: "категория | число" или "категория — число".
        Возвращает dict для EChartsConfig или None, если таблица не найдена.
        """
        if not rag_context or not rag_context.strip():
            return None

        chart_type = self._detect_chart_type(user_query)

        # Паттерн: категория | число  или  категория —/–/-/: число
        pattern = r"([^\n|—–:;]+?)\s*(?:\||—|–|-|:)\s*(\d+(?:[.,]\d+)?)"
        matches = re.findall(pattern, rag_context)

        if len(matches) < 2:
            # Попробуем табуляцию или двоеточие: категория: число
            # Категория не должна содержать точек (иначе цепляется "5. Гражданским...")
            pattern2 = r"([^.\n:]{3,60}?)\s*\\?\s*:\s*(\d+(?:[.,]\d+)?)"
            matches = re.findall(pattern2, rag_context)
            if len(matches) < 2:
                return None

        x_axis = []
        series_data = []
        for cat, val in matches:
            cat_clean = cat.strip().strip(" ,;")
            # Убираем нумерацию списка ("1) ", "1. ", "2) " и т.д.) в начале строки
            cat_clean = re.sub(r'^[\d\s]*[\.\)]\s*', '', cat_clean)
            cat_clean = cat_clean.strip()
            # Отсекаем мусор: слишком длинные строки, предложения с точками, нумерацию списков
            if len(cat_clean) < 5 or len(cat_clean) > 60:
                continue
            if re.search(r'(предоставляется|устанавливается|соответствии|федеральн|закон|отпуск|продолжительност|дополнительн|оплачиваем)', cat_clean, re.I):
                continue
            if '.' in cat_clean or ';' in cat_clean:  # Отсекаем нумерованные пункты ("5. Гражданским...")
                continue
            try:
                num = float(val.replace(",", "."))
            except ValueError:
                continue
            x_axis.append(cat_clean)
            series_data.append(num)

        if len(series_data) < 2:
            return None

        # Определяем единицу измерения из контекста
        unit = self._detect_unit_from_context(rag_context)

        title = self._make_title("data", user_query) or "Данные из контекста"
        logger.info(f"✅ [CHART] Таблица извлечена из RAG-контекста: x_axis={x_axis}, series_data={series_data}")

        return {
            "title": title,
            "chart_type": chart_type,
            "x_axis": x_axis,
            "series_name": "Значение",
            "series_data": series_data,
            "unit": unit,
        }

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
            
            # Если разница в порядках > 5000 — это разные таблицы
            if min_val > 0 and max_val / min_val > 5000:
                return False, (
                    f"Данные имеют слишком большой разброс значений "
                    f"(мин={min_val}, макс={max_val}, соотношение={max_val/min_val:.0f}x). "
                    f"График с такими данными будет некорректным."
                )
        
        return True, ""
    
    def validate_and_parse(self, raw_content: str, rag_context: str = "", user_query: str = "") -> dict:
        """Валидирует JSON через контракт Pydantic и страхует фронтенд от краша"""
        content = raw_content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        
        # Перед валидацией проверяем, а не пустой ли JSON прислала модель из-за отсутствия данных
        try:
            test_obj = json.loads(content)
            # Если ключевые поля пустые — значит данных в тексте не было
            if test_obj.get("series_data") == []:
                return {
                    "type": "chart_error",
                    "message": "⚠️ В базе знаний нет числовых данных для построения этого графика. Пожалуйста, уточните запрос или проверьте контекст.",
                    "raw": content
                }
        except Exception:
            pass

        content = self._sanitize_json(content)

        try:
            fixed_obj = json.loads(content)
            fixed_obj = self._expand_short_numbers(fixed_obj)
            content = json.dumps(fixed_obj, ensure_ascii=False)
        except Exception:
            pass
            
        try:
            validated_data = EChartsConfig.model_validate_json(content)
            validated_dict = validated_data.model_dump()
            
            return {
                "type": "chart",
                "data": validated_dict
            }
        except Exception as e:
            logger.warning(f"⚠️ [PYDANTIC VALIDATION ERROR]: {e}")
            # 1. Пытаемся достать данные из сырого JSON модели
            extracted = self._extract_from_raw_json(content, user_query)

            # 2. Если парсер вернул что-то, проверяем — не мусор ли это?
            if extracted:
                # Очищаем категории от остатков нумерации списков ("1) ", "1. ")
                for i, cat in enumerate(extracted.get("x_axis", [])):
                    if isinstance(cat, str):
                        extracted["x_axis"][i] = re.sub(r'^[\d\s]*[\.\)]\s*', '', cat).strip()
                # Мусорные признаки: длинные категории (как целые предложения) или стоп-слова
                x_str = " ".join(str(x) for x in extracted.get("x_axis", [])).lower()
                is_garbage = (
                    len(x_str) > 80 or  # слишком длинные категории
                    any(w in x_str for w in ["source", "type", "предоставляется", "устанавливается", "указом", "федеральн"])
                )

                if is_garbage and rag_context:
                    # Мусор — даем шанс умному парсеру контекста
                    context_extracted = self._extract_table_from_context(rag_context, user_query)
                    if context_extracted:
                        logger.info(f"♻️ [CHART] Raw-парсер дал мусор, используем таблицу из контекста")
                        return {"type": "chart", "data": context_extracted}
                    else:
                        # Если и контекст не спас — возвращаем ошибку (уйдет в текст)
                        return {"type": "chart_error", "message": "Данные не найдены", "raw": content[:200]}

                # Если не мусор — восстанавливаем unit из контекста и возвращаем
                if extracted.get("unit") is None and rag_context:
                    inferred = self._detect_unit_from_context(rag_context)
                    if inferred:
                        extracted["unit"] = inferred
                logger.info(f"✅ [CHART] Универсальный парсер сработал: {extracted}")
                return {"type": "chart", "data": extracted}

            # 3. Если raw-парсер вообще ничего не нашел
            if rag_context:
                context_extracted = self._extract_table_from_context(rag_context, user_query)
                if context_extracted:
                    return {"type": "chart", "data": context_extracted}

            return {"type": "chart_error", "message": "Не удалось извлечь структуру данных", "raw": content[:200]}
