import os, glob, json, logging, asyncio
import re
import time
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from pathlib import Path
import urllib.parse

# ========== ИМПОРТЫ ДЛЯ РЕРАНКЕРА ==========
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. НАСТРОЙКА ЛОГГЕРА (Уровень INFO для ROG терминала)
logger = logging.getLogger(__name__)

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Определяем корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ПРЯМОЙ ПУТЬ к новой базе (e5-base), которую создал скрипт s5_base.py
MODEL_PATH = os.path.join(BASE_DIR, "hf_cache", "multilingual-e5-base")

# Проверка для подстраховки
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ ВНИМАНИЕ: Папка {MODEL_PATH} не найдена! Проверь запуск s5_base.py")
else:
    print(f"🚀 ENGINE UPGRADE: Использую e5-base из {MODEL_PATH}")

PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_index")

# 2. ШАБЛОНЫ ПРОМПТОВ (Стиль "Строгий Аналитик")
qa_prompt_str = (
"Ты — эксперт ФНС России. Отвечай на русском языке.\n"
"---------------------\n"
"БАЗА ЗНАНИЙ:\n"
"{context_str}\n"
"---------------------\n"
"ПРАВИЛА:\n"
"1. НИКОГДА не начинай ответ со слов 'ТЕКСТ', 'ОТВЕТ', 'ИНФОРМАЦИЯ' или аналогичных. Начинай сразу с сути ответа.\n"
"2. Форматирование: заголовки через ###, списки: 1. для инструкций, • для перечислений, **жирный** для терминов и ФИО.\n"
"3. Тип ответа:\n"
"   - ИНСТРУКЦИЯ (как, инструкция, регистрация) → заголовок + список. Без таблиц и графиков.\n"
"   - ГРАФИК (график, диаграмма, круговая, столбчатая, гистограмма) → [CHART_JSON] с валидным JSON. Если в базе знаний нет числовых данных — напиши 'БАЗА_ПУСТА: Нет данных для визуализации'.\n"
"   - ТАБЛИЦА (таблица, топ, рейтинг) → заголовок + Markdown-таблица.\n"
"   - ТЕКСТ → во всех остальных случаях.\n"
"4. Форматы графиков (одинарные скобки {}):\n"
"   bar/line: [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"xAxis\":{\"type\":\"category\",\"data\":[\"А\",\"Б\"]},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"bar\",\"data\":[45678,38234]}]}[/CHART_JSON]\n"
"   pie: [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"series\":[{\"type\":\"pie\",\"data\":[{\"value\":45678,\"name\":\"Категория 1\"}]}]}[/CHART_JSON]\n"
"5. Если вопрос не по теме ФНС → 'БАЗА_ПУСТА: Я эксперт по вопросам ФНС'\n"
"6. Если спрашивают 'кто ты' → 'Я ИИ-помощник по базе знаний ФНС'\n"
"\n"
"ВОПРОС: {query_str}\n"
"ОТВЕТ:"
)

qa_prompt = PromptTemplate(qa_prompt_str)

# 3. ЕДИНЫЕ НАСТРОЙКИ
Settings.embed_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH, 
    device="cpu")
    
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="yagpt5:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1", 
        "num_ctx": 2048, 
        "temperature": 0, 
        "num_gpu": 99, 
        "num_predict":600,
        "seed": 42}
)

# ========== РЕРАНКЕР (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ) ==========
class SimpleReranker:
    def __init__(self):
        model_path = os.path.join(BASE_DIR, "reanker")
        self.device = "cpu"  # Принудительно CPU, т.к. у тебя нет GPU в докере
        logger.info(f"🔄 Загрузка реранкера из {model_path} на {self.device.upper()}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Кэш для результатов
        self.cache = {}
        self.cache_maxsize = 100
        
        print("✅ Реранкер загружен")
    
    def rerank(self, query, documents, top_k=3):
        # Кэширование
        cache_key = hash(query + "".join(d[:100] for d in documents[:5]))
        if cache_key in self.cache:
            print("⚡ Использую кэш реранкера")
            return self.cache[cache_key]
        
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = []
        
        # Маленький batch для CPU
        batch_size = 4
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                    max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Для BGE-reranker
                batch_scores = torch.sigmoid(outputs.logits).squeeze(-1).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        result = ranked[:top_k]
        
        # Сохраняем в кэш
        if len(self.cache) > self.cache_maxsize:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = result
        
        return result  
    
# ========== КАСТОМНЫЙ QUERY ENGINE С РЕРАНКЕРОМ ==========
class RerankedEngine:
    def __init__(self, index, reranker, qa_prompt, initial_top_k=10, final_top_k=3):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.reranker = reranker
        self.qa_prompt = qa_prompt
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
    
    def query(self, query_text):
        # Получаем initial_top_k документов
        nodes = self.retriever.retrieve(query_text)
        docs = [n.node.get_content() for n in nodes]
        
        # Реранжируем
        reranked = self.reranker.rerank(query_text, docs, top_k=self.final_top_k)
        
        # Собираем обратно nodes с новыми скорами
        from llama_index.core.schema import NodeWithScore
        reranked_nodes = []
        for doc_text, score in reranked:
            for n in nodes:
                if n.node.get_content() == doc_text:
                    reranked_nodes.append(NodeWithScore(node=n.node, score=score))
                    break
        
        # Генерируем ответ
        from llama_index.core.response_synthesizers import get_response_synthesizer
        synthesizer = get_response_synthesizer(text_qa_template=self.qa_prompt, streaming=True)
        return synthesizer.synthesize(query_text, reranked_nodes)
    
    async def aquery(self, query_text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, query_text)

# 4. ИНИЦИАЛИЗАЦИЯ (Загрузка в RAM)
logger.info("🧠 Загрузка векторного индекса в 32 ГБ RAM...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Инициализируем реранкер
reranker = SimpleReranker()

# Создаем query engine с реранкером
query_engine = RerankedEngine(index, reranker, qa_prompt, initial_top_k=10, final_top_k=3)
print("✅ Query engine с реранкером готов")

async def run_in_thread(func, *args):
    """Запускает функцию в отдельном потоке, не блокируя event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

# 🛡️ Глобальная блокировка для защиты GPU/Ollama от параллельных запросов
rag_gpu_lock = asyncio.Lock()
MAX_QUEUE_WAIT_SECONDS = 60.0  # Максимальное время ожидания в очереди (секунд)

async def get_ai_streaming_response(query_text: str):
    """
    Асинхронный генератор для стриминга ответа от RAG + LLM.
    
    Особенности:
    - Защита от параллельных запросов через asyncio.Lock
    - Таймаут ожидания в очереди
    - Полные замеры времени (RAG, генерация, ожидание)
    - Автоматический вывод фото руководителей ФНС
    """
    start_time = time.time()
    
    try:
        # 🛡️ ШАГ 0: ЗАХВАТ ОЧЕРЕДИ С ТАЙМАУТОМ
        async with asyncio.timeout(MAX_QUEUE_WAIT_SECONDS):
            async with rag_gpu_lock:
                queue_wait_time = time.time() - start_time
                if queue_wait_time > 1.0:
                    logger.warning(f"⏳ Долгое ожидание очереди: {queue_wait_time:.1f} сек для '{query_text[:50]}...'")
                
                logger.info(f"🚀 [LOCK ACQUIRED] Начало обработки: '{query_text[:50]}...' (очередь: {queue_wait_time:.1f} сек)")
                
                query_lower = query_text.lower()
                
                try:
                    # 🔍 ШАГ 1: RAG ПОИСК (под lock'ом, т.к. вызывает Ollama)
                    rag_start = time.time()
                    response = await asyncio.to_thread(query_engine.query, query_text)
                    rag_time = time.time() - rag_start
                    logger.info(f"🔍 RAG поиск + реранкинг выполнен за {rag_time:.2f} сек")
                    
                    nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                    logger.info(f"🧩 После реранкинга: {len(nodes)} чанков")
                    
                    # 🔍 Отладочный вывод топ-3 score
                    for i, node in enumerate(nodes[:3]):
                        score = node.score if hasattr(node, 'score') else 0.0
                        title = node.node.metadata.get('title', 'без названия')[:50]
                        logger.info(f"   📊 Топ-{i+1}: score={score:.3f}, title={title}")
                    
                    # Собираем источники из всех nodes (без фильтрации по порогу)
                    sources = []
                    seen_urls = set()
                    has_real_context = len(nodes) > 0
                    
                    for node in nodes:
                        u = node.node.metadata.get("source_url")
                        t = node.node.metadata.get("title", "Источник")
                        if u and u not in seen_urls:
                            sources.append({"url": u, "title": t})
                            seen_urls.add(u)
                    
                    # Отправка метаданных
                    yield json.dumps({
                        "type": "metadata",
                        "sources": sources,
                        "has_answer": has_real_context,
                        "image": None
                    }, ensure_ascii=False) + "\n"
                    
                    # 💬 ШАГ 2: ГЕНЕРАЦИЯ ТЕКСТА (стриминг)
                    gen_start = time.time()
                    token_count = 0
                    full_response_text = ""
                    
                    if has_real_context and hasattr(response, "response_gen"):
                        for token in response.response_gen:
                            full_response_text += token
                            token_count += 1
                            yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"
                    else:
                        yield json.dumps({"type": "text", "content": "Информация не найдена."}, ensure_ascii=False) + "\n"
                        total_time = time.time() - start_time
                        logger.info(f"⏱️ [НЕТ ОТВЕТА] Запрос '{query_text[:50]}...' - {total_time:.2f} сек")
                        return
                    
                    gen_time = time.time() - gen_start
                    logger.info(f"💬 Генерация {token_count} токенов за {gen_time:.2f} сек ({token_count/gen_time:.1f} ток/сек)")
                    
                    # 📸 ШАГ 3: ФОТО-ФИЛЬТР (для руководителей ФНС)
                    final_photo = None
                    resp_lower = full_response_text.lower()
                    
                    # Проверка на "База пуста" или оффтоп
                    if "база_пуста" in resp_lower or "эксперт только по вопросам фнс" in resp_lower:
                        logger.info("🚫 ВОПРОС ВНЕ ТЕМЫ: Блокировка вывода фото")
                    else:
                        # Блокировка фото для таблиц и графиков
                        is_chart = "[chart_json]" in resp_lower
                        is_table = "|---" in resp_lower or "| :---" in resp_lower or (resp_lower.count("|") > 10)
                        
                        if is_chart or is_table:
                            logger.info("📊 Данные в таблице/графике: блокируем фото")
                        else:
                            img_folder = Path("images_cache")
                            if img_folder.exists():
                                # Словарь фото
                                photo_map = {f.stem.lower(): f.name for f in img_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
                                
                                # Поиск по фамилии в тексте ответа
                                for name_key, filename in photo_map.items():
                                    surname = name_key.split()[0]
                                    if surname in resp_lower:
                                        final_photo = filename
                                        break
                                
                                # Если не нашли, берем из метаданных лучшего чанка
                                if not final_photo and nodes:
                                    best_img_score = -1
                                    for node in nodes[:3]:  # Только топ-3
                                        score = node.score if hasattr(node, 'score') else 0.0
                                        img_name = node.node.metadata.get("local_img")
                                        if img_name and score > best_img_score:
                                            if (img_folder / img_name).exists():
                                                best_img_score = score
                                                final_photo = img_name
                    
                    # Вывод фото
                    if final_photo and "ии-помощник" not in resp_lower:
                        encoded_name = urllib.parse.quote(final_photo)
                        photo_md = f"\n\n![photo](/images/{encoded_name})"
                        yield json.dumps({"type": "text", "content": photo_md}, ensure_ascii=False) + "\n"
                        logger.info(f"📸 УСПЕХ: Выведено {final_photo}")
                    
                    # ⏱️ ФИНАЛЬНЫЙ ЗАМЕР
                    total_time = time.time() - start_time
                    other_time = total_time - queue_wait_time - rag_time - gen_time
                    
                    logger.info(f"⏱️ [ИТОГО] Запрос '{query_text[:50]}...' - {total_time:.2f} сек")
                    logger.info(f"   ├─ Ожидание очереди: {queue_wait_time:.2f} сек")
                    logger.info(f"   ├─ RAG поиск + реранкинг: {rag_time:.2f} сек")
                    logger.info(f"   ├─ Генерация: {gen_time:.2f} сек ({token_count} токенов)")
                    logger.info(f"   └─ Прочее: {other_time:.2f} сек")
                    
                except Exception as e:
                    logger.error(f"❌ ОШИБКА в обработке: {str(e)}", exc_info=True)
                    yield json.dumps({"type": "error", "content": f"Ошибка системы: {str(e)}"}) + "\n"
                    
    except TimeoutError:
        logger.error(f"⏰ Таймаут очереди: запрос '{query_text[:50]}...' ждал > {MAX_QUEUE_WAIT_SECONDS} сек")
        yield json.dumps({
            "type": "error",
            "content": "Сервер временно перегружен, попробуйте позже."
        }, ensure_ascii=False) + "\n"
        return
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)
        yield json.dumps({"type": "error", "content": f"Критическая ошибка: {str(e)}"}) + "\n"
        return
        
#Бот мультик в телеграмм
async def get_ai_response_full(query_text: str):
    """Специально для ТГ-бота: возвращает словарь для красивого вывода"""
    try:
        response = await query_engine.aquery(query_text)
        nodes = response.source_nodes
        
        # Собираем ссылки (Score > 0.65)
        sources = []
        seen_urls = set()
        first_img = None
        
        for node in nodes:
            score = node.score if hasattr(node, 'score') else 0.0
            if score > 0.65:
                u = node.node.metadata.get("source_url")
                t = node.node.metadata.get("title", "Инструкция")
                img = node.node.metadata.get("local_img")
                
                if u and u not in seen_urls:
                    sources.append((u, t)) # Кортеж для InlineKeyboardButton
                    seen_urls.add(u)
                if img and not first_img:
                    first_img = img
        
        return {
            "answer": str(response),
            "sources": sources,
            "image": first_img
        }
    except Exception as e:
        logger.error(f"❌ Ошибка в RAG-движке: {e}")
        return {"answer": "В моих регламентах про это ни слова, бро.", "sources": [], "image": None}