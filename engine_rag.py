import os
import re
import json
import logging
import asyncio
import time
import urllib.parse
import numpy as np
from pathlib import Path
from typing import List, Optional, Any, Tuple

# LlamaIndex Core
from llama_index.core import (
    StorageContext, 
    load_index_from_storage, 
    Settings,
    PromptTemplate, 
    QueryBundle
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.embeddings import BaseEmbedding

# Модели и Дополнения
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

# Гибридный поиск и лингвистика
from rank_bm25 import BM25Okapi
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, models


# ========== BM25 ==========
try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ rank-bm25 не установлен. Ставь: pip install rank-bm25")

# ========== ЛОГИРОВАНИЕ ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_rag")

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ========== ПУТИ (константы) ==========
BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = BASE_DIR / "hf_cache" / "ru-en-RoSBERTa"
PERSIST_DIR = BASE_DIR / "fns_rag_graph_final"
IMG_FOLDER  = BASE_DIR / "images_cache"
EMPLOYEES_FILE = BASE_DIR / "employees.txt"

if not MODEL_PATH.exists():
    logger.warning(f"⚠️ Папка модели не найдена: {MODEL_PATH}")
else:
    logger.info(f"🚀 Использую RoSBERTa из {MODEL_PATH}")

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
logger.info(f"✨ Загрузка эмбеддера RoSBERTa на CPU: {MODEL_PATH}")


class SberRoSBERTaEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        # Собираем модель вручную, чтобы ЖЕСТКО задать CLS pooling
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), 
            pooling_mode='cls' # 🔥 Вот твоя заветная строчка, теперь она тут
        )
        self._model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], 
            device=device
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        # Твои префиксы Сбера
        return self._model.encode(f"search_query: {query}").tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        # Твои префиксы Сбера
        return self._model.encode(f"search_document: {text}").tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)



Settings.embed_model = SberRoSBERTaEmbedding(
    model_path=str(MODEL_PATH), 
    device="cpu"
)


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

def _load_employees() -> List[Tuple[str, List[str]]]:
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
        variations = [name]
        if parts:
            variations.append(parts[0])
        if len(parts) >= 2:
            variations.append(f"{parts[0]} {parts[1]}")
        result.append((emp_full, [v for v in variations if len(v) > 5]))
    return result

_EMPLOYEES: List[Tuple[str, List[str]]] = _load_employees()
logger.info(f"👥 Загружено сотрудников: {len(_EMPLOYEES)}")

_EMPTY_RESPONSE_RE = re.compile(
    r"база_пуста|эксперт только по вопросам фнс|информация отсутствует",
    re.IGNORECASE,
)

_TITLE_PREFIXES = ("Указ №", "ФЗ №", "Приказ №", "Письмо №")


class RerankedEngine:
    def __init__(self, index: Any, qa_prompt: Any, initial_top_k: int = 30, final_top_k: int = 5):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.qa_prompt = qa_prompt
        self.final_top_k = final_top_k
        self.stemmer = SnowballStemmer("russian")
        
        self.bm25 = None
        self.all_nodes = []
        self.node_map = {} 
        
        try:
            logger.info("🔧 Инициализация гибридного движка...")
            for doc_id in index.docstore.docs:
                doc = index.docstore.docs[doc_id]
                if hasattr(doc, 'get_content'):
                    self.all_nodes.append(doc)
                    self.node_map[doc.node_id] = doc
            
            logger.info(f"📚 База знаний: {len(self.all_nodes)} нод загружено")
            
            start_time = time.time()
            tokenized_corpus = [self._tokenize(n.get_content()) for n in self.all_nodes]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ Индекс BM25 готов за {time.time() - start_time:.2f} сек")
        except Exception as e:
            logger.error(f"❌ Ошибка BM25: {e}", exc_info=True)
        
        try:
            reranker_path = str(Path("/app/reranker"))
            logger.info(f"🧠 Загрузка BGE-Reranker из {reranker_path}...")
            self.reranker = SentenceTransformerRerank(
                model=reranker_path, 
                top_n=self.final_top_k,
                device="cpu"
            )
            logger.info("✅ Реранкер готов (CPU)")
        except Exception as e:
            logger.error(f"⚠️ Реранкер OFF: {e}")
            self.reranker = None

        # 🔥 УДАЛИЛ self.synthesizer — теперь создаём в _sync_query динамически

    def _tokenize(self, text: str) -> List[str]:
        if not text: return []
        clean_text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', text.lower())
        return [self.stemmer.stem(word) for word in clean_text.split() if len(word) > 2]
    
    def _reciprocal_rank_fusion(self, vector_nodes, bm25_scores, k=60):
        scores = {}
        for rank, node in enumerate(vector_nodes):
            scores[node.node_id] = scores.get(node.node_id, 0) + 1.0 / (k + rank + 1)
        
        bm25_indices = np.argsort(bm25_scores)[::-1][:30]
        valid_bm25_count = 0
        for rank, idx in enumerate(bm25_indices):
            if bm25_scores[idx] <= 0: continue
            node_id = self.all_nodes[idx].node_id
            scores[node_id] = scores.get(node_id, 0) + 1.0 / (k + rank + 1)
            valid_bm25_count += 1
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = [NodeWithScore(node=self.node_map[nid], score=scores[nid]) for nid in sorted_ids if nid in self.node_map]
        
        logger.info(f"📊 RRF Hybrid: Vector({len(vector_nodes)}) + BM25({valid_bm25_count}) -> Total({len(result)})")
        return result
    
    def _sync_query(self, query_text: str):
        overall_start = time.time()
        logger.info(f"🔎 [QUERY START]: {query_text}")
        
        # 1. Вектор
        initial_nodes = self.retriever.retrieve(query_text)
        
        # 2. Гибрид
        if self.bm25 is not None and initial_nodes:
            bm25_scores = self.bm25.get_scores(self._tokenize(query_text))
            if np.any(bm25_scores > 0): 
                initial_nodes = self._reciprocal_rank_fusion(initial_nodes, bm25_scores)
        
        # ЛОГИ ТОП-10 ПОСЛЕ ПОИСКА
        print("\n" + "="*20 + " HYBRID TOP-10 " + "="*20)
        for i, n in enumerate(initial_nodes[:10]):
            nid = n.node.metadata.get('id', 'N/A')
            print(f"Rank {i+1}: [{n.score:.4f}] ID: {nid}")
        print("="*55 + "\n")

        if not initial_nodes:
            logger.warning("‼️ Ничего не найдено")
            return None

        # 3. Реранкинг
        if self.reranker and initial_nodes:
            logger.info("🧪 Запуск реранкинга (CPU)...")
            rt_start = time.time()
            nodes_to_rerank = initial_nodes[:10]
            final_nodes = self.reranker.postprocess_nodes(nodes_to_rerank, query_bundle=QueryBundle(query_text))
            logger.info(f"⏱ Реранкинг 10 нод -> {len(final_nodes)} занял {time.time() - rt_start:.2f} сек")
        else:
            final_nodes = initial_nodes[:self.final_top_k]
            logger.info("⚠️ Реранкер пропущен")

        # ФИНАЛЬНЫЙ СПИСОК ПЕРЕД LLM
        print("\n" + "!"*20 + " FINAL TOP-5 FOR LLM " + "!"*20)
        for i, n in enumerate(final_nodes):
            nid = n.node.metadata.get('id', 'N/A')
            words = len(n.node.get_content().split())
            print(f"TOP-{i+1}: [{n.score:.4f}] ID: {nid} (~{words} слов)")
        print("!"*61 + "\n")

        # 🔥 ДИНАМИЧЕСКИЙ ВЫБОР РЕЖИМА
        q_low = query_text.lower()
        visual_keywords = [
    "граф", "диагр", "кругов", "столбчат", "гистогр", 
    "табл", "схем", "покаж", "вывед", "рисуй", "иллюстр",
    "центральн", "ца фнс", " ца ", "кто такой" # пробелы вокруг "ца", чтобы не ловить слово "цапля"
]

        is_visual = any(kw in q_low for kw in visual_keywords)
        
        if is_visual:
            current_mode = ResponseMode.COMPACT
            logger.info("⚡ Выбран режим: COMPACT (графики/таблицы)")
        else:
            current_mode = ResponseMode.TREE_SUMMARIZE
            logger.info("🌲 Выбран режим: TREE_SUMMARIZE (синтез)")
        
        # Создаём синтезатор под выбранный режим
        synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=current_mode
        )
        
       
        # 4. Генерация
        final_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Базовая инструкция
        instruction = "ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ, ИСПОЛЬЗУЯ ТОЛЬКО БАЗУ ЗНАНИЙ. "
        # Защита от LaTeX
        latex_fix = "Пиши дроби простыми цифрами (1/4). ЗАПРЕЩЕНО использовать LaTeX и символы '$'."
        
        forced_query = f"{query_text}\n\nВАЖНО: {instruction} {latex_fix}"

        
        logger.info("🚀 Отправка в Ollama...")
        response = synthesizer.synthesize(forced_query, final_nodes)
        
        total_time = time.time() - overall_start
        logger.info(f"🏁 [DONE] Итоговое время: {total_time:.2f} сек")
        
        return response

    async def aquery(self, query_text: str):
        return await asyncio.to_thread(self._sync_query, query_text)
              
# ========== ИНИЦИАЛИЗАЦИЯ ==========
logger.info("🧠 Загрузка векторного индекса...")
storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
index = load_index_from_storage(storage_context)

query_engine = RerankedEngine(
    index=index,
    qa_prompt=qa_prompt,
    initial_top_k=30,
    final_top_k=5,
)
logger.info("✅ Query engine готов к работе")

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (осталось без изменений) ==========
def _collect_sources(nodes: list, max_sources: int = 3) -> list:
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
                "score": round(float(node.score), 4) if hasattr(node, "score") else None,
            })
        seen_urls.add(url)
    return sources

def _find_photo(resp_lower: str, nodes: list) -> Optional[str]:
    for filename, variations in _EMPLOYEES:
        for variation in variations:
            if variation in resp_lower:
                return filename

    if nodes and hasattr(nodes[0], "node"):
        best_node = nodes[0]
        img_name = best_node.node.metadata.get("local_img")
        if img_name:
            photo_path = IMG_FOLDER / img_name
            if photo_path.exists():
                logger.info(f"📸 Найдена картинка в лучшем чанке: {img_name}")
                return img_name

    return None

# ========== ОСНОВНАЯ ФУНКЦИЯ ДЛЯ API ==========
async def get_ai_streaming_response(query_text: str):
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

        for i, node in enumerate(nodes[:5]):
            if hasattr(node, 'node'):
                node_id = node.node.metadata.get('id', 'unknown')
                score = node.score if hasattr(node, 'score') else 0.0
                logger.info(f"   📊 Чанк {i+1}: id={node_id[:40]}, score={score:.4f}")

        sources = _collect_sources(nodes)
        logger.info(f"🧩 Источников для фронта: {len(sources)}")

        yield json.dumps({
            "type": "metadata",
            "sources": sources,
            "has_answer": has_real_context,
        }, ensure_ascii=False) + "\n"

        if not has_real_context or not hasattr(response, "response_gen") or response.response_gen is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        gen_start = time.time()
        tokens: List[str] = []

        for token in response.response_gen:
            tokens.append(token)
            yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"

        gen_time = time.time() - gen_start
        token_count = len(tokens)
        if gen_time > 0:
            logger.info(f"💬 {token_count} токенов за {gen_time:.2f} сек ({token_count / gen_time:.1f} ток/сек)")
        else:
            logger.info(f"💬 {token_count} токенов за {gen_time:.2f} сек")

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