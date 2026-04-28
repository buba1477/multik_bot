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
MODEL_PATH = os.path.join(BASE_DIR, "hf_cache", "multilingual-e5-large")

# Проверка для подстраховки
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ ВНИМАНИЕ: Папка {MODEL_PATH} не найдена! Проверь запуск s5_base.py")
else:
    print(f"🚀 ENGINE UPGRADE: Использую e5-base из {MODEL_PATH}")

PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_graph_final")

# 2. ШАБЛОНЫ ПРОМПТОВ (Стиль "Строгий Аналитик")
qa_prompt_str = (
"Ты — эксперт ФНС России.\n"
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

"\n"
"ВОПРОС: {query_str}\n\n"
"ОТВЕТ ЭКСПЕРТА:" 
)

qa_prompt = PromptTemplate(qa_prompt_str)

# 1. Скопируй класс прямо над настройками
class PrefixedEmbedding(HuggingFaceEmbedding):
    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding("query: " + query)
    async def _aget_text_embedding(self, text: str):
        return await super()._aget_text_embedding("passage: " + text)
    async def _aget_query_embedding(self, query: str):
        return await super()._aget_query_embedding("query: " + query)

# 2. Примени его в Settings
Settings.embed_model = PrefixedEmbedding(
    model_name=MODEL_PATH, 
    device="cpu"
)
    
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="yagpt5_4096:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1", 
        "num_ctx": 4096, 
        "temperature": 0.1, 
        "num_gpu": 33, 
        "num_predict": 1024,
        "seed": 42}
)


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_rag")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 1. УСКОРЕННЫЙ РЕРАНКЕР С УМНЫМ КЭШЕМ ==========
class SimpleReranker:
    def __init__(self):
        model_path = os.path.join(BASE_DIR, "reanker")
        self.device = "cpu"
        logger.info(f"🔄 Загрузка реранкера из {model_path} на {self.device.upper()}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.cache = {}
        self.cache_maxsize = 200
        logger.info("✅ Реранкер успешно загружен")

    def rerank(self, query, documents, top_k=4):
        if not documents:
            return []

        # Дедупликация входящих текстов
        unique_documents = list(dict.fromkeys(documents))
        
        # Умный кэш (хэшируем всё)
        cache_key = hash(query + "".join(unique_documents))
        if cache_key in self.cache:
            logger.info("⚡ Использую кэш реранкера")
            return self.cache[cache_key]
        
        pairs = [[query, doc] for doc in unique_documents]
        scores = []
        batch_size = 10 # Твой ROG съест это быстро
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            # max_length=256 для ускорения на CPU (по совету)
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                    max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = torch.sigmoid(outputs.logits).view(-1).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
        
        ranked = sorted(zip(unique_documents, scores), key=lambda x: x[1], reverse=True)
        result = ranked[:top_k]
        
        if len(self.cache) >= self.cache_maxsize:
            self.cache.clear()
        self.cache[cache_key] = result
        
        return result

# ========== 2. КАСТОМНЫЙ ДВИЖОК: ВЕРСИЯ "ТОТАЛЬНЫЙ РЕРАНК" ==========
class RerankedEngine:
    def __init__(self, index, reranker, qa_prompt, initial_top_k=20, final_top_k=5):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.reranker = reranker
        self.qa_prompt = qa_prompt
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
    
    def query(self, query_text):
        # 1. Первичный поиск (Берем все 20 чанков)
        initial_nodes = self.retriever.retrieve(query_text)
        if not initial_nodes:
            return "Контекст не найден."
        
        # logger.info(f"🚀 Получено {initial_nodes} ")

        # 2. РЕРАНКЕР: Теперь он видит всё, что нашел E5 без исключений
        docs_for_rerank = [n.node.get_content() for n in initial_nodes]
        reranked_results = self.reranker.rerank(query_text, docs_for_rerank, top_k=self.final_top_k)
        
        

        # 3. СБОРКА КОНТЕКСТА С ГЛУБОКОЙ СКЛЕЙКОЙ
        from llama_index.core.schema import NodeWithScore, TextNode
        final_nodes = []
        final_seen_roots = set() 
        
        for i, (doc_text, score) in enumerate(reranked_results):
            # Ищем, какая именно нода из 20-ти соответствует этому тексту
            matched_n = next((n for n in initial_nodes if n.node.get_content() == doc_text), None)
            if not matched_n: continue
            
            n_id = matched_n.node.metadata.get('id', '')
            # Извлекаем корень (79фз_st46_p3 -> 79фз_st46)
            # Надо сделать:
            if "community" in str(n_id):
                # Для обзоров корнем является сам ID — склейки с соседями не будет
                art_root = str(n_id) 
            else:
                # Для законов оставляем старую логику склейки по статьям
                art_root = re.sub(r'_(?:p|part|st)?\d+$', '', str(n_id))

            # Если мы уже обработали эту статью (например, через другой её чанк) - скипаем
            if art_root in final_seen_roots:
                continue

            # 🔥 ЛОГИКА РАСКРЫТИЯ (Для Топ-1 и Топ-2 результатов реранкера)
            if i < 2:
                # Берем все куски этой статьи, которые попали в изначальные 20
                all_related = [n for n in initial_nodes if n.node.metadata.get('id', '').startswith(art_root)]
                all_related.sort(key=lambda x: x.node.metadata.get('id', ''))
                
                try:
                    # Находим индекс победителя в списке чанков этой статьи
                    current_idx = next(idx for idx, n in enumerate(all_related) if n.node.metadata.get('id') == n_id)
                    # Берем соседей вокруг него (-2 / +3)
                    start_idx = max(0, current_idx - 2)
                    end_idx = min(len(all_related), current_idx + 3)
                    related = all_related[start_idx:end_idx]
                    
                    logger.info(f"🧱 ГЛУБОКАЯ СКЛЕЙКА: {art_root} (взято {len(related)} фрагм. вокруг {n_id})")
                except StopIteration:
                    related = [matched_n]

                full_text = "\n\n".join([p.node.get_content() for p in related])
                all_urls = list(set([p.node.metadata.get('url') for p in related if p.node.metadata.get('url')]))
                
                new_node = TextNode(text=full_text, metadata={**matched_n.node.metadata, 'combined_urls': all_urls})
                final_nodes.append(NodeWithScore(node=new_node, score=score))
                final_seen_roots.add(art_root)
            
            # Для остальных (Топ-3) просто добавляем чанк как есть
            else:
                final_nodes.append(NodeWithScore(node=matched_n.node, score=score))
                final_seen_roots.add(art_root)

        # 4. ГЕНЕРАЦИЯ
        from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
        
        # Определяем режим (для юридических тестов лучше оставить COMPACT, если чанков мало)
        selected_mode = ResponseMode.COMPACT if len(final_nodes) <= 2 else ResponseMode.TREE_SUMMARIZE
        logger.info(f"🚀 ResponseMode: {selected_mode}")
        synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt, 
            streaming=True,
            response_mode=selected_mode
        )
        
        # Порог 0.1, чтобы не потерять важные, но сложные ответы
        final_nodes = [n for n in final_nodes if n.score > 0.1] 
        
        if not final_nodes:
            final_nodes = [initial_nodes[0]]
        
        # logger.info(f"🚀 final_nodes: {final_nodes}")
        
        forced_query = f"{query_text}\n\nОТВЕТЬ СТРОГО НА РУССКОМ ЯЗЫКЕ!"


        return synthesizer.synthesize(forced_query, final_nodes)

    async def aquery(self, query_text):
        return await asyncio.to_thread(self.query, query_text)


# 4. ИНИЦИАЛИЗАЦИЯ (Загрузка в RAM)
logger.info("🧠 Загрузка векторного индекса в 32 ГБ RAM...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Инициализируем реранкер
reranker = SimpleReranker()

# Создаем query engine с реранкером
query_engine = RerankedEngine(index, reranker, qa_prompt, initial_top_k=15, final_top_k=5)
print("✅ Query engine с реранкером готов")

async def run_in_thread(func, *args):
    """Запускает функцию в отдельном потоке, не блокируя event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)
    
MAX_QUEUE_WAIT_SECONDS = 60.0  # Максимальное время ожидания (уже не используется, но оставил для совместимости)

async def get_ai_streaming_response(query_text: str):
    """
    Асинхронный генератор для стриминга ответа от RAG + LLM.
    
    Особенности:
    - ПОЛНОСТЬЮ АСИНХРОННЫЙ (без блокировок)
    - Полные замеры времени (RAG, генерация)
    - Автоматический вывод фото руководителей ФНС
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Начало обработки: '{query_text[:50]}...'")
        
        query_lower = query_text.lower()
        
        # 🔍 ШАГ 1: RAG ПОИСК (асинхронно, без блокировки)
        rag_start = time.time()
        # Важно: query_engine.query должен быть асинхронным или обёрнут в asyncio.to_thread
        response = await asyncio.to_thread(query_engine.query, query_text)
        rag_time = time.time() - rag_start
        logger.info(f"🔍 RAG поиск + реранкинг выполнен за {rag_time:.2f} сек")
        
        nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        logger.info(f"🧩 После реранкинга: {len(nodes)} чанков")
        
        # 🔍 Отладочный вывод топ-3 score
        for i, node in enumerate(nodes[:5]):
            score = node.score if hasattr(node, 'score') else 0.0
            logger.info(f"   📊 Топ-{i+1}: score={score:.3f}")
            logger.info(f"      metadata: {node.node.metadata}")
            id = node.node.metadata.get('id', 'без названия')
            title = node.node.metadata.get('title', 'без названия')[:50]
            # logger.info(f"   📊 Топ-{i+1}: score={score:.3f}, id={id}, title={title}")
        
        # Собираем источники
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
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
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
                # Загружаем список сотрудников
                EMPLOYEES_FILE = Path("employees.txt")
                if EMPLOYEES_FILE.exists():
                    with open(EMPLOYEES_FILE, 'r', encoding='utf-8') as f:
                        EMPLOYEES = [line.strip() for line in f if line.strip()]
                else:
                    # Fallback список
                    EMPLOYEES = [
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

                img_folder = Path("images_cache")
                            
                # Поиск по фамилии
                for name_key in EMPLOYEES:
                    surname = name_key.split()[0].lower()
                    if surname in resp_lower:
                        final_photo = name_key
                        break
                # Если не нашли, берем из метаданных лучшего чанка
                if not final_photo and nodes:
                    best_img_score = -1
                    for node in nodes[:3]:  # Только топ-3
                        score = node.score if hasattr(node, 'score') else 0.0
                        img_name = node.node.metadata.get("local_img")
                        if img_name and score > best_img_score:
                            if (img_folder / img_name).exists() and node.node.metadata.get("type") != 'person':
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
        other_time = total_time - rag_time - gen_time
        
        logger.info(f"⏱️ [ИТОГО] Запрос '{query_text[:50]}...' - {total_time:.2f} сек")
        logger.info(f"   ├─ RAG поиск + реранкинг: {rag_time:.2f} сек")
        logger.info(f"   ├─ Генерация: {gen_time:.2f} сек ({token_count} токенов)")
        logger.info(f"   └─ Прочее: {other_time:.2f} сек")
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА в обработке: {str(e)}", exc_info=True)
        yield json.dumps({"type": "error", "content": f"Ошибка системы: {str(e)}"}) + "\n"
        
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