import os
import json
import logging
import asyncio
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional, Any, AsyncGenerator

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_rag")

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Определяем корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ПРЯМОЙ ПУТЬ к эмбеддеру
MODEL_PATH = os.path.join(BASE_DIR, "hf_cache", "multilingual-e5-large")

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ ВНИМАНИЕ: Папка {MODEL_PATH} не найдена!")
else:
    print(f"🚀 Использую e5-large из {MODEL_PATH}")

PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_graph_final")

# ========== ПРОМПТ (полная версия из оригинала) ==========
qa_prompt_str = (
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

qa_prompt = PromptTemplate(qa_prompt_str)

# ========== ЭМБЕДДЕР ==========
class PrefixedEmbedding(HuggingFaceEmbedding):
    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)
    
    def _get_query_embedding(self, query: str):
        print(f"🔍 [SEARCH] query: {query[:100]}...")
        return super()._get_query_embedding("query: " + query)

Settings.embed_model = PrefixedEmbedding(model_name=MODEL_PATH, device="cpu")

# ========== LLM (OLLAMA) ==========
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="yagpt5_fns:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1",      # 👈 модель всегда в памяти
        "num_ctx": 4096, 
        "temperature": 0, 
        "num_gpu": 33, 
        "num_predict": 1024,
        "seed": 42
    }
)

# ========== ДВИЖОК С РЕРАНКОМ ==========
class RerankedEngine:
    def __init__(self, index: Any, qa_prompt: Any, initial_top_k: int = 10, final_top_k: int = 5):
        """
        initial_top_k: сколько кандидатов тянет E5
        final_top_k: сколько оставляет BGE-реранкер
        """
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.qa_prompt = qa_prompt
        self.final_top_k = final_top_k
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        reranker_path = os.path.join(base_dir, "reranker")  # 👈 исправил опечатку
        
        # Инициализируем реранкер
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
        # --- ШАГ 1: ПОИСК E5 ---
        initial_nodes = self.retriever.retrieve(query_text)
        if not initial_nodes:
            logger.warning("❌ Нет результатов от E5")
            return None
        
        # Логируем TOP-10 от E5
        print("\n" + "="*35 + " E5 TOP-10 " + "="*35)
        for idx, n in enumerate(initial_nodes[:10]):
            print(f"   [{idx+1:2d}] {n.node.metadata.get('id', 'N/A')[:30]:30s} | Score: {n.score:.4f}")

        # --- ШАГ 2: РЕРАНКИНГ (BGE) ---
        if self.reranker:
            # Передаем первые 10 кандидатов на реранкинг
            nodes_to_rerank = initial_nodes[:10]
            query_bundle = QueryBundle(query_text)
            final_nodes = self.reranker.postprocess_nodes(nodes_to_rerank, query_bundle=query_bundle)
            
            # Логируем TOP-5 после реранкинга
            print("\n" + "!"*35 + " BGE TOP-5 (FINAL) " + "!"*35)
            for idx, n in enumerate(final_nodes):
                print(f"   [TOP-{idx+1:2d}] {n.node.metadata.get('id', 'N/A')[:30]:30s} | Score: {n.score:.4f}")
            print("!"*88 + "\n")
        else:
            # Если реранкер не загрузился, берем первые final_top_k
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
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = RerankedEngine(
    index=index,
    qa_prompt=qa_prompt,
    initial_top_k=10,  # E5 тянет 15 кандидатов
    final_top_k=5       # BGE оставляет 5 лучших
)
print("✅ Query engine готов к работе")

# ========== ОСНОВНАЯ ФУНКЦИЯ ДЛЯ API ==========
async def get_ai_streaming_response(query_text: str):
    """
    Основной API-эндпоинт для стримингового ответа.
    Возвращает генератор JSON-строк с типами:
    - metadata: служебная информация (источники, img)
    - text: токены ответа
    - end: сигнал завершения
    - error: ошибка
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Начало обработки запроса: '{query_text[:100]}...'")
        
        # Получаем ответ от движка
        response = await query_engine.aquery(query_text)
        
        # Проверка на пустой ответ
        if response is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return
        
        # Извлекаем найденные ноды (чанки)
        nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        has_real_context = len(nodes) > 0
        
        logger.info(f"🧩 Найдено релевантных чанков: {len(nodes)}")
        
        # Логируем топ-5 чанков с их скорами
        for i, node in enumerate(nodes[:5]):
            score = node.score if hasattr(node, 'score') else 0.0
            node_id = node.node.metadata.get('id', 'unknown') if hasattr(node, 'node') else 'unknown'
            logger.info(f"   📊 Топ-{i+1}: score={score:.4f}, id={node_id[:40]}")
        
        # Собираем уникальные источники (URL + заголовки)
        sources = []
        seen_urls = set()
        first_image = nodes[0].node.metadata.get("local_img", "") if nodes else ""
        
        for node in nodes:
            if not hasattr(node, 'node'):
                continue
            url = node.node.metadata.get("source_url")
            title = node.node.metadata.get("title", "Источник")
            if url and url not in seen_urls:
                sources.append({"url": url, "title": title})
                seen_urls.add(url)
        
        # Отправляем метаданные клиенту
        yield json.dumps({
            "type": "metadata",
            "sources": sources,
            "has_answer": has_real_context,
            "image": None,
            "img": first_image,
        }, ensure_ascii=False) + "\n"
        
        # --- ГЕНЕРАЦИЯ ТОКЕНОВ ---
        if not has_real_context or not hasattr(response, 'response_gen'):
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return
        
        gen_start = time.time()
        token_count = 0
        full_response_text = ""
        
        for token in response.response_gen:
            full_response_text += token
            token_count += 1
            yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"
        
        gen_time = time.time() - gen_start
        logger.info(f"💬 Сгенерировано {token_count} токенов за {gen_time:.2f} сек ({token_count/gen_time:.1f} ток/сек)")
        
        # --- ПОСТ-ОБРАБОТКА: ФИЛЬТР ФОТОГРАФИЙ ---
        final_photo = None
        resp_lower = full_response_text.lower()
        
        # Проверяем, что ответ не содержит признаков пустоты
        is_empty_response = any(phrase in resp_lower for phrase in [
            "база_пуста", "эксперт только по вопросам фнс", "информация отсутствует"
        ])
        
        if not is_empty_response:
            # Определяем, не график и не таблица
            is_chart = "[chart_json]" in resp_lower
            is_table = "|---" in resp_lower or "| :---" in resp_lower or (resp_lower.count("|") > 10)
            
            if not is_chart and not is_table:
                # Загружаем список сотрудников из файла или используем дефолтный
                employees_file = Path("employees.txt")
                if employees_file.exists():
                    with open(employees_file, 'r', encoding='utf-8') as f:
                        employees = [line.strip() for line in f if line.strip()]
                else:
                    employees = [
                        "Егоров Даниил Вячеславович.jpeg",
                        "Петрушин Андрей Станиславович.jpg",
                        "Бударин Андрей Владимирович.jpeg",
                        "Бондарчук Светлана Леонидовна.jpg",
                        "Сатин Дмитрий Станиславович.jpg",
                        "Шиналиев Тимур Николаевич.jpg",
                        "Шепелева Юлия Вячеславовна.jpg",
                        "Бациев Виктор Валентинович.jpg",
                        "Колесников Виталий Григорьевич.jpg",
                        "Егоричев Александр Валерьевич.jpg",
                        "Чекмышев Константин Николаевич.jpg"
                    ]
                
                # Проверяем, упоминается ли кто-то из сотрудников в ответе
                for name_key in employees:
                    surname = name_key.split()[0].lower()
                    if surname in resp_lower:
                        final_photo = name_key
                        break
                
                # Если не нашли по фамилии, берем фотку из первого релевантного чанка
                if not final_photo and nodes:
                    img_folder = Path("images_cache")
                    best_img_score = -1
                    for node in nodes[:3]:
                        img_name = node.node.metadata.get("local_img")
                        if img_name and (img_folder / img_name).exists():
                            score = node.score if hasattr(node, 'score') else 0.0
                            node_type = node.node.metadata.get("type", "")
                            if score > best_img_score and node_type != 'person':
                                best_img_score = score
                                final_photo = img_name
        
        # Отправляем фотографию, если нашли
        if final_photo and "ии-помощник" not in resp_lower:
            encoded_name = urllib.parse.quote(final_photo)
            photo_md = f"\n\n![photo](/images/{encoded_name})"
            yield json.dumps({"type": "text", "content": photo_md}, ensure_ascii=False) + "\n"
            logger.info(f"📸 Добавлено фото: {final_photo}")
        
        # Сигнал завершения
        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Полное время обработки запроса: {total_time:.2f} сек")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка в get_ai_streaming_response: {str(e)}", exc_info=True)
        yield json.dumps({"type": "error", "content": f"Ошибка сервера: {str(e)}"}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"


# ========== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (без стриминга) ==========
async def get_ai_response_full(query_text: str):
    """
    Получение полного ответа без стриминга (для тестов или синхронных вызовов)
    """
    try:
        logger.info(f"📝 Синхронный запрос: '{query_text[:100]}...'")
        response = await query_engine.aquery(query_text)
        
        if not response or not hasattr(response, 'source_nodes'):
            return {
                "answer": "БАЗА_ПУСТА: Информация не найдена.",
                "sources": [],
                "image": None
            }
        
        # Собираем полный текст из стрима
        full_text = ""
        if hasattr(response, 'response_gen'):
            for token in response.response_gen:
                full_text += token
        else:
            full_text = str(response)
        
        # Собираем источники
        sources = []
        seen = set()
        first_img = None
        
        for node in response.source_nodes[:5]:
            if not hasattr(node, 'node'):
                continue
            url = node.node.metadata.get("source_url")
            title = node.node.metadata.get("title", "Источник")
            img = node.node.metadata.get("local_img")
            
            if url and url not in seen:
                sources.append({"url": url, "title": title})
                seen.add(url)
            if img and not first_img:
                first_img = img
        
        return {
            "answer": full_text,
            "sources": sources,
            "image": first_img
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка в get_ai_response_full: {e}", exc_info=True)
        return {
            "answer": "В моих регламентах про это ни слова, бро.",
            "sources": [],
            "image": None
        }


# ========== ТЕСТОВЫЙ ЗАПУСК (если файл запущен напрямую) ==========
if __name__ == "__main__":
    async def test():
        print("\n🧪 ТЕСТОВЫЙ ЗАПУСК")
        query = "Расскажи про увольнение за утрату доверия"
        print(f"Вопрос: {query}\n")
        
        async for chunk in get_ai_streaming_response(query):
            try:
                data = json.loads(chunk)
                if data["type"] == "text":
                    print(data["content"], end="")
                elif data["type"] == "metadata":
                    print(f"\n📚 Источников: {len(data.get('sources', []))}")
            except:
                pass
        print("\n\n✅ Тест завершен")
    
    asyncio.run(test())