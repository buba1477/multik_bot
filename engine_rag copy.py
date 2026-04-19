import os, glob, json, logging, asyncio
import re
import time
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from pathlib import Path
from sentence_transformers import CrossEncoder
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

PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_index")

# 2. ШАБЛОНЫ ПРОМПТОВ (Стиль "Строгий Аналитик")
qa_prompt_str = (
"Ты — эксперт ФНС России. Отвечай СТРОГО на русском языке.\n"
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
        "temperature": 0, 
        "num_gpu": 33, 
        "num_predict": 1024,
        "seed": 42}
)


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_rag")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 1. УСКОРЕННЫЙ РЕРАНКЕР (SENTENCE-TRANSFORMERS) ==========
class SimpleReranker:
    def __init__(self):
        model_path = os.path.join(BASE_DIR, "reanker")
        self.device = "cpu"
        logger.info(f"🔄 Загрузка реранкера BGE-V2-M3 из {model_path} на {self.device.upper()}...")
        
        # CrossEncoder сам подгрузит tokenizer и model, используя указанный device
        self.model = CrossEncoder(
            model_path, 
            device=self.device, 
            max_length=256  # Оптимально для скорости на CPU
        )
        
        self.cache = {}
        self.cache_maxsize = 200
        logger.info("✅ Реранкер успешно загружен через CrossEncoder")

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []

        # 1. Дедупликация (чтобы не считать одно и то же дважды)
        unique_documents = list(dict.fromkeys(documents))
        
        # 2. Проверка кэша
        cache_key = hash(query + "".join(unique_documents))
        if cache_key in self.cache:
            logger.info("⚡ Использую кэш реранкера")
            return self.cache[cache_key]
        
        # 3. Подготовка пар для модели
        pairs = [[query, doc] for doc in unique_documents]
        
        # 4. Предикт (predict сам делает батчинг и применяет активацию)
        # batch_size=10-20 оптимально для ROG на CPU
        scores = self.model.predict(
            pairs, 
            batch_size=20, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Если пришел один скор (float), превращаем в список
        if isinstance(scores, (float, int)):
            scores = [scores]

        # 5. Ранжирование
        ranked = sorted(zip(unique_documents, scores), key=lambda x: x[1], reverse=True)
        result = ranked[:top_k]
        
        # 6. Обновление кэша
        if len(self.cache) >= self.cache_maxsize:
            self.cache.clear()
        self.cache[cache_key] = result
        
        return result
    
# ========== 2. КАСТОМНЫЙ ДВИЖОК: ДИНАМИЧЕСКАЯ РАСПАКОВКА ТОП-1 И ТОП-2 ==========
class RerankedEngine:
    def __init__(self, index, reranker, qa_prompt, initial_top_k=20, final_top_k=4):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.reranker = reranker
        self.qa_prompt = qa_prompt
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
    
    def _extract_art_root(self, node_id: str) -> str:
        """Универсальное извлечение корня статьи"""
        # Убираем суффиксы вида _p1, _p8¹, _part2, _st5
        return re.sub(r'_(?:p|part|st)?\d+[¹²³⁰-⁹]*$', '', node_id)
    
    def _normalize_id(self, node_id: str) -> str:
        """Нормализует ID для сравнения (убирает спецсимволы)"""
        return re.sub(r'[¹²³⁰-⁹]', '', node_id)
    
    def query(self, query_text):
        # 1. Первичный поиск
        initial_nodes = self.retriever.retrieve(query_text)
        if not initial_nodes:
            return "Контекст не найден."
        
        logger.info(f"🚀 Получено {len(initial_nodes)} чанков от ретривера")
        
        # 2. Группируем по корням статей
        from collections import defaultdict
        articles = defaultdict(list)
        
        for n in initial_nodes:
            n_id = n.node.metadata.get('id', '')
            art_root = self._extract_art_root(n_id)
            articles[art_root].append(n)
        
        logger.info(f"📚 Найдено {len(articles)} уникальных статей")
        
        # 3. Для реранкера берем топ-2 чанка из каждой статьи
        docs_for_rerank = []
        nodes_for_rerank = []
        
        for art_root, nodes_list in articles.items():
            nodes_list.sort(key=lambda x: x.score, reverse=True)
            for node in nodes_list[:2]:
                docs_for_rerank.append(node.node.get_content())
                nodes_for_rerank.append(node)
        
        logger.info(f"🔄 Отправляем реранкеру {len(docs_for_rerank)} чанков")
        
        # 4. Реранкинг
        reranked_results = self.reranker.rerank(query_text, docs_for_rerank, top_k=self.final_top_k * 2)
        
        # 5. СБОРКА КОНТЕКСТА С ЖЕСТКИМ ЛИМИТОМ
        from llama_index.core.schema import NodeWithScore, TextNode
        
        final_nodes = []
        processed_articles = set()
        
        # 🔥 НОВЫЕ КОНСТАНТЫ ДЛЯ ОГРАНИЧЕНИЯ
        MAX_CHUNKS_PER_ARTICLE = 2  # Максимум 2 чанка на статью (для не-топ1)
        MAX_TOTAL_CHUNKS = 5        # Максимум 5 чанков ВСЕГО в контексте
        total_chunks_used = 0       # Счетчик использованных чанков
        
        # Определяем тип вопроса
        is_numeric_question = any(word in query_text.lower() for word in [
            'срок', 'сколько', 'какое число', 'период', 'дней', 'часов', 
            '21 день', '30 дней', '15 дней', '7 дней'
        ])
        
        for i, (doc_text, score) in enumerate(reranked_results):
            # 🔥 ПРОВЕРКА ЛИМИТА
            if total_chunks_used >= MAX_TOTAL_CHUNKS:
                logger.info(f"⏹️ Достигнут лимит чанков ({MAX_TOTAL_CHUNKS}), останавливаю сбор контекста")
                break
            
            # Находим соответствующую ноду
            matched_n = None
            for n in nodes_for_rerank:
                if n.node.get_content() == doc_text:
                    matched_n = n
                    break
            
            if not matched_n:
                continue
            
            n_id = matched_n.node.metadata.get('id', '')
            art_root = self._extract_art_root(n_id)
            
            # Если статья уже обработана - пропускаем
            if art_root in processed_articles:
                continue
            
            # 🎯 Для числовых вопросов - берем ровно 1 чанк
            if is_numeric_question:
                if total_chunks_used + 1 <= MAX_TOTAL_CHUNKS:
                    final_nodes.append(NodeWithScore(node=matched_n.node, score=score))
                    processed_articles.add(art_root)
                    total_chunks_used += 1
                    logger.info(f"📊 ЧИСЛОВОЙ ВОПРОС: беру точный чанк {n_id} (всего {total_chunks_used}/{MAX_TOTAL_CHUNKS})")
                else:
                    logger.info(f"⏹️ Лимит достигнут, пропускаю {n_id}")
                continue
            
            # Для остальных вопросов - склеиваем, но с ограничениями
            all_related = articles.get(art_root, [])
            all_related.sort(key=lambda x: x.node.metadata.get('id', ''))
            
            # 🔥 ГЛАВНОЕ ИЗМЕНЕНИЕ: Для топ-1 статьи берем ВСЕ чанки (до лимита)
            if i == 0:  # Самая релевантная статья по версии реранкера
                chunks_to_take = min(len(all_related), MAX_TOTAL_CHUNKS - total_chunks_used)
                if chunks_to_take > 0:
                    related = all_related[:chunks_to_take]
                    logger.info(f"🏆 ТОП-1 СТАТЬЯ: беру {len(related)} из {len(all_related)} чанков (вся статья)")
                else:
                    continue
            else:
                # Для остальных статей - берем 1-2 чанка как раньше
                if len(all_related) <= MAX_CHUNKS_PER_ARTICLE:
                    related = all_related
                else:
                    # Пытаемся найти соседей
                    current_id_normalized = self._normalize_id(n_id)
                    current_idx = None
                    
                    for idx, n in enumerate(all_related):
                        n_id_normalized = self._normalize_id(n.node.metadata.get('id', ''))
                        if n_id_normalized == current_id_normalized:
                            current_idx = idx
                            break
                    
                    if current_idx is not None:
                        # Берем 1 до и 1 после (всего до 3 чанков)
                        start_idx = max(0, current_idx - 1)
                        end_idx = min(len(all_related), current_idx + 2)
                        related = all_related[start_idx:end_idx]
                        # Обрезаем до MAX_CHUNKS_PER_ARTICLE
                        if len(related) > MAX_CHUNKS_PER_ARTICLE:
                            related = related[:MAX_CHUNKS_PER_ARTICLE]
                        logger.info(f"🧱 СКЛЕЙКА: {art_root} (взято {len(related)} фрагм. вокруг {n_id})")
                    else:
                        # Не нашли - берем топ чанки
                        related = all_related[:MAX_CHUNKS_PER_ARTICLE]
                        logger.warning(f"⚠️ Не найден индекс для {n_id}, беру {len(related)} фрагм.")
            
            # 🔥 ПРОВЕРЯЕМ ЛИМИТ ПЕРЕД ДОБАВЛЕНИЕМ
            if total_chunks_used + len(related) > MAX_TOTAL_CHUNKS:
                # Берем только то, что влезает
                chunks_to_take = MAX_TOTAL_CHUNKS - total_chunks_used
                if chunks_to_take <= 0:
                    logger.info(f"⏹️ Лимит достигнут, пропускаю статью {art_root}")
                    continue
                related = related[:chunks_to_take]
                logger.info(f"✂️ Обрезано до {chunks_to_take} чанков из-за лимита")
            
            # Собираем текст
            full_text_parts = []
            all_urls = set()
            
            for p in related:
                full_text_parts.append(p.node.get_content())
                if p.node.metadata.get('url'):
                    all_urls.add(p.node.metadata.get('url'))
            
            full_text = "\n\n".join(full_text_parts)
            
            new_node = TextNode(
                text=full_text,
                metadata={
                    **matched_n.node.metadata,
                    'combined_urls': list(all_urls),
                    'original_chunks': len(related)
                }
            )
            final_nodes.append(NodeWithScore(node=new_node, score=score))
            processed_articles.add(art_root)
            total_chunks_used += len(related)
            logger.info(f"🚀 СТАТЬЯ РАСКРЫТА: {art_root} ({len(related)} чанков, всего {total_chunks_used}/{MAX_TOTAL_CHUNKS})")
        
        # 6. Если ничего не собрали - берем топ от ретривера (но не больше 3 чанков)
        if not final_nodes and initial_nodes:
            final_nodes = [NodeWithScore(node=initial_nodes[i].node, score=initial_nodes[i].score) 
                          for i in range(min(3, len(initial_nodes)))]
            logger.info(f"⚠️ Фолбек: взял {len(final_nodes)} чанков из ретривера")
        
        # 7. ГЕНЕРАЦИЯ
        from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
        
        # Для числовых вопросов используем COMPACT режим
        numeric_keywords = ['срок', 'сколько', 'дней', 'часов', 'период', 'какое число']
        is_numeric = any(word in query_text.lower() for word in numeric_keywords)
        is_simple = is_numeric or any(word in query_text.lower() for word in ['график', 'диаграмма', 'таблица', 'кто', 'какой', 'фио'])
        
        selected_mode = ResponseMode.COMPACT if is_simple else ResponseMode.TREE_SUMMARIZE
        logger.info(f"🚀 ТИП selected_mode: {selected_mode} (numeric={is_numeric})")
        
        synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=selected_mode,
            use_async=True
        )
        
        forced_query = f"{query_text}\n\nОТВЕТЬ СТРОГО НА РУССКОМ ЯЗЫКЕ!"
        
        # Логируем что попало в контекст
        logger.info(f"📊 В контексте {len(final_nodes)} нод, всего чанков: {total_chunks_used}")
        for i, node in enumerate(final_nodes[:3]):
            preview = node.node.text[:200].replace('\n', ' ')
            logger.info(f"   Нода {i+1}: score={node.score:.3f}, preview={preview}...")
        
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
query_engine = RerankedEngine(index, reranker, qa_prompt, initial_top_k=20, final_top_k=4)
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
        for i, node in enumerate(nodes[:4]):
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