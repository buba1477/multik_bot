import os
import re
import json
import logging
import asyncio
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional, Any, Tuple

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate, QueryBundle

# ========== ЛОГИРОВАНИЕ ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_rag")

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ========== ПУТИ (константы) ==========
BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = BASE_DIR / "hf_cache" / "multilingual-e5-large"
PERSIST_DIR = BASE_DIR / "fns_rag_graph_final"
IMG_FOLDER  = BASE_DIR / "images_cache"
EMPLOYEES_FILE = BASE_DIR / "employees.txt"

if not MODEL_PATH.exists():
    logger.warning(f"⚠️ Папка модели не найдена: {MODEL_PATH}")
else:
    logger.info(f"🚀 Использую E5 из {MODEL_PATH}")

# ========== ПРОМПТ ==========
_QA_PROMPT_STR = (
    "Ты — ведущий эксперт ФНС России. Отвечай СТРОГО на русском языке.\n"
    "---------------------\n"
    "БАЗА ЗНАНИЙ:\n"
    "{context_str}\n"
    "---------------------\n"
    "ПРАВИЛА:\n"
    "1. ЧИСТЫЙ ТЕКСТ: ЗАПРЕЩЕНО выводить техническую информацию (ID, ЧАНК, СОДЕРЖАНИЕ, ТЕКСТ) и служебные заголовки (Указ №1574 и т.д.). НИКОГДА не начинай ответ со слов 'ОТВЕТ', 'ИНФОРМАЦИЯ' или аналогичных. Пиши как живой эксперт.\n"
    "2. Форматирование: заголовки через ###, списки: 1. для инструкций, • для перечислений, **жирный** для терминов и ФИО.\n"
    "3. Тип ответа:\n"
    "   - ИНСТРУКЦИЯ (как, инструкция, регистрация) → заголовок + список. Без таблиц и графиков.\n"
    "   - ГРАФИК (график, диаграмма, круговая, круг, доли, столбчатая, гистограмма, динамика) → [CHART_JSON] с валидным JSON. Если в базе знаний нет числовых данных — напиши 'БАЗА_ПУСТА: Нет данных для визуализации'.\n"
    "   - ТАБЛИЦА (таблица, топ, рейтинг) → заголовок + Markdown-таблица.\n"
    "   - ТЕКСТ → во всех остальных случаях.\n"
    "4. Форматы графиков (одинарные скобки {}):\n"
    "   bar/line: [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"xAxis\":{\"type\":\"category\",\"data\":[\"А\",\"Б\"]},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"bar\",\"data\":[45678,38234]}]}[/CHART_JSON]\n"
    "   pie: [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"series\":[{\"type\":\"pie\",\"data\":[{\"value\":45678,\"name\":\"Категория 1\"}]}]}[/CHART_JSON]\n"
    "5. СТРОГОСТЬ: Используй ТОЛЬКО базу знаний. Отвечай уверенно, без 'вероятно' и 'предположительно'. Если инфы нет — 'БАЗА_ПУСТА: Информация отсутствует'.\n"
    "6. ТЕСТЫ: Если есть варианты ответов — выбери ОДИН самый точный по базе. Не обобщай!\n"
    "7. Если вопрос не по теме — 'БАЗА_ПУСТА: Я эксперт по вопросам ФНС'.\n"
    "8. ТОТАЛЬНЫЙ СИНТЕЗ: Если информация по вопросу разбросана по разным чанкам, объедини в один ответ.\n"
    "9. ССЫЛКИ: В конце ответа укажи статьи закона.\n\n"
    "ВОПРОС: {query_str}\n\n"
    "ОТВЕТ ЭКСПЕРТА:"
)

qa_prompt = PromptTemplate(_QA_PROMPT_STR)

_FORCED_QUERY_SUFFIX = "\n\nВАЖНО: ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ, ИСПОЛЬЗУЯ ТОЛЬКО БАЗУ ЗНАНИЙ."

# ========== ЭМБЕДДЕР ==========
logger.info(f"✨ Загрузка эмбеддера 5E-large на CPU: {MODEL_PATH}")

# ========== ЭМБЕДДЕР ==========
class PrefixedEmbedding(HuggingFaceEmbedding):
    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)
    
    def _get_query_embedding(self, query: str):
        print(f"🔍 [SEARCH] query: {query[:100]}...")
        return super()._get_query_embedding("query: " + query)

Settings.embed_model = PrefixedEmbedding(model_name=str(MODEL_PATH), device="cpu")


# ========== LLM (OLLAMA) ==========
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="yagpt5_fns:latest",
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1",
        "num_ctx": 4096,
        "temperature": 0,
        "num_gpu": 33,
        "num_predict": 1024,
        "seed": 42,
    }
)

# ========== СОТРУДНИКИ (загружаем один раз при старте) ==========
_DEFAULT_EMPLOYEES = [
    "Егоров Даниил Вячеславович.jpeg",
    "Петрушин Андрей Станиславович.jpg",
    "Бударин Андрей Владимирович.jpeg",  # ← ИСПРАВЛЕНО: была латинская B
    "Бондарчук Светлана Леонидовна.jpg",
    "Сатин Дмитрий Станиславович.jpg",
    "Шиналиев Тимур Николаевич.jpg",
    "Шепелева Юлия Вячеславовна.jpg",
    "Бациев Виктор Валентинович.jpg",
    "Колесников Виталий Григорьевич.jpg",
    "Егоричев Александр Валерьевич.jpg",
    "Чекмышев Константин Николаевич.jpg"
]

def _load_employees() -> List[Tuple[str, List[str]]]:
    """
    Возвращает список (filename, [вариации имени]) один раз при старте.
    Вариации: полное ФИО, только фамилия, фамилия+имя.
    """
    if EMPLOYEES_FILE.exists():
        with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        logger.warning("employees.txt не найден, используется встроенный список")
        raw = _DEFAULT_EMPLOYEES

    result: List[Tuple[str, List[str]]] = []
    for emp_full in raw:
        name = emp_full.rsplit(".", 1)[0].lower()
        parts = name.split()
        variations = [name]                                     # Полное ФИО
        if parts:
            variations.append(parts[0])                         # Фамилия
        if len(parts) >= 2:
            variations.append(f"{parts[0]} {parts[1]}")        # Фамилия Имя
        # Оставляем только вариации длиннее 5 символов (защита от ложных срабатываний)
        result.append((emp_full, [v for v in variations if len(v) > 5]))
    return result

# Кэш сотрудников — загружается ровно один раз
_EMPLOYEES: List[Tuple[str, List[str]]] = _load_employees()
logger.info(f"👥 Загружено сотрудников: {len(_EMPLOYEES)}")

# Заранее скомпилированный шаблон для проверки «пустого» ответа
_EMPTY_RESPONSE_RE = re.compile(
    r"база_пуста|эксперт только по вопросам фнс|информация отсутствует",
    re.IGNORECASE,
)

_TITLE_PREFIXES = ("Указ №", "ФЗ №", "Приказ №", "Письмо №")

# ========== ДВИЖОК С РЕРАНКОМ (BACK TO BASICS) ==========
class RerankedEngine:
    def __init__(self, index: Any, qa_prompt: Any, initial_top_k: int = 10, final_top_k: int = 5):
        """
        initial_top_k: сколько кандидатов тянет E5/RoSBERTa
        final_top_k: сколько оставляет BGE-реранкер
        """
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.qa_prompt = qa_prompt
        self.final_top_k = final_top_k
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        reranker_path = os.path.join(base_dir, "reranker")
        
        logger.info(f"🧠 Загрузка BGE-Reranker из {reranker_path}")
        try:
            self.reranker = SentenceTransformerRerank(
                model=reranker_path, 
                top_n=self.final_top_k,
                device="cpu"
            )
            logger.info("✅ Реранкер успешно загружен")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки реранкера: {e}")
            self.reranker = None

        self.synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=ResponseMode.COMPACT
        )
    
    def _sync_query(self, query_text: str):
        # --- ШАГ 1: ПОИСК ---
        initial_nodes = self.retriever.retrieve(query_text)
        if not initial_nodes:
            logger.warning("❌ Нет результатов поиска")
            return None
        
        # Логируем TOP-10
        print("\n" + "="*35 + " SEARCH TOP-10 " + "="*35)
        for idx, n in enumerate(initial_nodes[:10]):
            print(f"   [{idx+1:2d}] {n.node.metadata.get('id', 'N/A')[:30]:30s} | Score: {n.score:.4f}")

        # --- ШАГ 2: РЕРАНКИНГ (BGE) ---
        if self.reranker and initial_nodes:
            nodes_to_rerank = initial_nodes[:10]
            query_bundle = QueryBundle(query_text)
            final_nodes = self.reranker.postprocess_nodes(nodes_to_rerank, query_bundle=query_bundle)
            
            # Логируем TOP-5 после реранкинга
            print("\n" + "!"*35 + " BGE TOP-5 (FINAL) " + "!"*35)
            for idx, n in enumerate(final_nodes):
                print(f"   [TOP-{idx+1:2d}] {n.node.metadata.get('id', 'N/A')[:30]:30s} | Score: {n.score:.4f}")
            print("!"*88 + "\n")
        else:
            final_nodes = initial_nodes[:self.final_top_k]
            print(f"⚠️ Реранкер не используется, беру первые {self.final_top_k} результатов")

        # --- ШАГ 3: ГЕНЕРАЦИЯ ---
        final_nodes.sort(key=lambda x: x.score, reverse=True)
        forced_query = f"{query_text}\n\nВАЖНО: ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ, ИСПОЛЬЗУЯ ТОЛЬКО БАЗУ ЗНАНИЙ."
        return self.synthesizer.synthesize(forced_query, final_nodes)

    async def aquery(self, query_text: str):
        """Асинхронная обёртка для синхронного метода"""
        return await asyncio.to_thread(self._sync_query, query_text)


# ========== ИНИЦИАЛИЗАЦИЯ ==========
logger.info("🧠 Загрузка векторного индекса...")
storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
index = load_index_from_storage(storage_context)

query_engine = RerankedEngine(
    index=index,
    qa_prompt=qa_prompt,
    initial_top_k=10,
    final_top_k=5,
)
logger.info("✅ Query engine готов к работе")


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def _collect_sources(nodes: list, max_sources: int = 3) -> list:
    """Собирает дедуплицированные источники из нод."""
    sources = []
    seen_urls: set = set()
    for node in nodes:
        if len(sources) >= max_sources:
            break
        if not hasattr(node, "node"):
            continue
        url = node.node.metadata.get("source_url", "")
        if not url or url in seen_urls:
            continue
        title = node.node.metadata.get("title", "Источник").strip()
        if title.startswith(_TITLE_PREFIXES):
            title = title.split(". ", 1)[-1] if ". " in title else title
        sources.append({
                "url": url,
                "title": title[:100],
                # ОБЯЗАТЕЛЬНО оберни в float(), чтобы JSON не ругался
                "score": round(float(node.score), 4) if hasattr(node, "score") else None,
            })
        seen_urls.add(url)
    return sources


def _find_photo(resp_lower: str, nodes: list) -> Optional[str]:
    """
    Ищет фото: 
    1. Сначала по ФИО в тексте ответа (глобальный приоритет).
    2. Если нет ФИО — берем картинку СТРОГО из лучшего чанка (Топ-1).
    """
    # 1. Сначала чекаем ФИО (это святое, если в ответе Егоров — нужно его фото)
    for filename, variations in _EMPLOYEES:
        for variation in variations:
            if variation in resp_lower:
                return filename

    # 2. Если ФИО не нашли — смотрим ТОЛЬКО в первый чанк (Top-1)
    if nodes and hasattr(nodes[0], "node"):
        best_node = nodes[0]
        # Проверяем наличие картинки в метаданных лучшего чанка
        img_name = best_node.node.metadata.get("local_img")
        
        if img_name:
            photo_path = IMG_FOLDER / img_name
            if photo_path.exists():
                # Логируем, что взяли картинку из топа
                logger.info(f"📸 Найдена картинка в лучшем чанке: {img_name}")
                return img_name

    # Если в Топ-1 пусто — возвращаем None (никакого визуального шума)
    return None


# ========== ОСНОВНАЯ ФУНКЦИЯ ДЛЯ API ==========
async def get_ai_streaming_response(query_text: str):
    """
    Стриминговый API-эндпоинт.
    Yields JSON-строки с типами: metadata | text | end | error
    """
    start_time = time.time()

    try:
        logger.info(f"🚀 Запрос: '{query_text[:100]}...'")

        response = await query_engine.aquery(query_text)

        if response is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        nodes = response.source_nodes if hasattr(response, "source_nodes") else []
        has_real_context = bool(nodes)

        logger.info(f"🧩 Найдено чанков: {len(nodes)}")

        # Логируем топ-5 чанков для отладки
        for i, node in enumerate(nodes[:5]):
            if hasattr(node, 'node'):
                node_id = node.node.metadata.get('id', 'unknown')
                score = node.score if hasattr(node, 'score') else 0.0
                logger.info(f"   📊 Чанк {i+1}: id={node_id[:40]}, score={score:.4f}")

        # Источники (без дублирования img)
        sources = _collect_sources(nodes)
        logger.info(f"🧩 Источников для фронта: {len(sources)}")

        # Отправляем метаданные БЕЗ img (чтобы не дублировать)
        yield json.dumps({
            "type": "metadata",
            "sources": sources,
            "has_answer": has_real_context,
        }, ensure_ascii=False) + "\n"

        if not has_real_context or not hasattr(response, "response_gen") or response.response_gen is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        # Генерация токенов
        gen_start = time.time()
        tokens: List[str] = []

        for token in response.response_gen:
            tokens.append(token)
            yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"

        gen_time = time.time() - gen_start
        token_count = len(tokens)
        if gen_time > 0:
            logger.info(
                f"💬 {token_count} токенов за {gen_time:.2f} сек "
                f"({token_count / gen_time:.1f} ток/сек)"
            )
        else:
            logger.info(f"💬 {token_count} токенов за {gen_time:.2f} сек")

        # Подбор фото
        full_response_text = "".join(tokens)
        resp_lower = full_response_text.lower()

        is_empty = bool(_EMPTY_RESPONSE_RE.search(resp_lower))
        is_chart = "[chart_json]" in resp_lower
        is_table = "|---" in resp_lower or "| :---" in resp_lower or resp_lower.count("|") > 10

        if not is_empty and not is_chart and not is_table:
            final_photo = _find_photo(resp_lower, nodes)
            if final_photo and "ии-помощник" not in resp_lower:
                encoded = urllib.parse.quote(final_photo)
                yield json.dumps(
                    {"type": "text", "content": f"\n\n![photo](/images/{encoded})"},
                    ensure_ascii=False,
                ) + "\n"
                logger.info(f"📸 Добавлено фото: {final_photo}")

        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
        logger.info(f"⏱️ Итого: {time.time() - start_time:.2f} сек")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        yield json.dumps({"type": "error", "content": f"Ошибка сервера: {e}"}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"


# ========== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (без стриминга) ==========
async def get_ai_response_full(query_text: str) -> dict:
    """Полный ответ без стриминга (для тестов / синхронных вызовов)."""
    try:
        logger.info(f"📝 Синхронный запрос: '{query_text[:100]}...'")
        response = await query_engine.aquery(query_text)

        if not response or not hasattr(response, "source_nodes"):
            return {"answer": "БАЗА_ПУСТА: Информация не найдена.", "sources": [], "image": None}

        tokens: List[str] = []
        if hasattr(response, "response_gen") and response.response_gen is not None:
            for token in response.response_gen:
                tokens.append(token)
        else:
            tokens.append(str(response))

        sources = _collect_sources(response.source_nodes[:5])
        first_img: Optional[str] = None
        for n in response.source_nodes[:5]:
            if hasattr(n, "node") and n.node.metadata.get("local_img"):
                first_img = n.node.metadata.get("local_img")
                break

        return {"answer": "".join(tokens), "sources": sources, "image": first_img}

    except Exception as e:
        logger.error(f"❌ Ошибка в get_ai_response_full: {e}", exc_info=True)
        return {"answer": "В моих регламентах про это ни слова, бро.", "sources": [], "image": None}


# ========== ТЕСТОВЫЙ ЗАПУСК ==========
if __name__ == "__main__":
    async def test():
        print("\n🧪 ТЕСТОВЫЙ ЗАПУСК")
        query = "Расскажи про увольнение за утрату доверия"
        print(f"Вопрос: {query}\n")

        async for chunk in get_ai_streaming_response(query):
            try:
                data = json.loads(chunk)
                if data["type"] == "text":
                    print(data["content"], end="", flush=True)
                elif data["type"] == "metadata":
                    print(f"\n📚 Источников: {len(data.get('sources', []))}")
            except json.JSONDecodeError:
                pass
        print("\n\n✅ Тест завершен")

    asyncio.run(test())