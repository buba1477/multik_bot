import os
from .sources import collect_sources
import re
import json
import logging
import asyncio
import time
import urllib.parse
import numpy as np
import requests
from pathlib import Path
from typing import List, Optional, Any, Tuple

# LlamaIndex Core
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.embeddings import BaseEmbedding

# Модели и Дополнения
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

# Гибридный поиск и лингвистика
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, models

from ollama import ChatResponse
from datetime import datetime
import uuid
import pickle

from llama_index.core.schema import MetadataMode  # <--- ДОБАВЬ MetadataMode

# Импорт графиков и визуализации (для будущего использования в ECharts)
from .chart_engine import DynamicChartEngine
from qdrant_client import QdrantClient  # СТРОГО ТАК
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
    QueryBundle
)

from pydantic.v1 import PrivateAttr
# FIXME: Временный патч для исправления несовместимости Ollama SDK 0.4.x и LlamaIndex.
# LlamaIndex пытается записать 'usage' в ChatResponse, который это запрещает.
# Удалить, когда в llama-index-llms-ollama выйдет фикс.

from gigachat import GigaChat as GigaChatSDK
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
# ========== ЛОГИРОВАНИЕ ==========
from ..logger import logger as app_logger

logger = app_logger


def patched_setitem(self, key, value):
    try:
        # Пытаемся записать нормально
        object.__setattr__(self, key, value)
    except Exception:
        # Если Pydantic орет — просто забиваем болт на это поле
        pass

# Подменяем метод записи во всей библиотеке на лету
# ChatResponse.__setitem__ = patched_setitem


# ========== BM25 ==========
try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("⚠️ rank-bm25 не установлен. Ставь: pip install rank-bm25")


# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ========== ПУТИ (константы) ==========
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
MODEL_PATH = BASE_DIR / "hf_cache" / "FRIDA"
PERSIST_DIR = BASE_DIR / "fns_rag_graph_final"
IMG_FOLDER = BASE_DIR / "images_cache"
EMPLOYEES_FILE = BASE_DIR / "employees.txt"

if not MODEL_PATH.exists():
    logger.warning(f"⚠️ Папка модели не найдена: {MODEL_PATH}")
else:
    logger.info(f"🚀 Использую RoSBERTa из {MODEL_PATH}")

_QA_PROMPT_STR = """Отвечай ТОЛЬКО на основании предоставленного КОНТЕКСТА.

Отвечай только на русском языке.

Не используй внутренние знания модели, память и внешнюю информацию.

Не выдумывай, отвечай только по контексту

Не путай норму гражданин (впервые поступающий на государсвенную гражданскую службу) и граджанский служащий (уже на службе)

ФОРМАТ ОТВЕТА:
Ответ ВСЕГДА должен начинаться строго с фразы:
**Ответ:** [короткий ответ норма контекста]

КОНТЕКСТ:
{context_str}

Вопрос пользователя:
{query_str}

"""

qa_prompt = PromptTemplate(_QA_PROMPT_STR)


# ========== ЭМБЕДДЕР ==========
logger.info(f"✨ Загрузка эмбеддера FRIDA на CPU: {MODEL_PATH}")


class SberRoSBERTaEmbedding(BaseEmbedding):
    _model: Any = PrivateAttr()

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"✨ Загрузка эмбеддера FRIDA на {device}")
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode='cls'
        )
        self._model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=device
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(f"search_query: {query}").tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
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
API_KEY_GIGACHAT = os.getenv("API_KEY_GIGACHAT", "")

# Инициализируем наш новый изолированный движок
chart_engine = DynamicChartEngine(ollama_url=OLLAMA_HOST)

# API GigaChat
# class LlamaGigaChat(CustomLLM):
#     context_window: int = 8096
#     num_output: int = 512
#     model_name: str = "GigaChat-2-Max"
#     # 🔥 Жестко берем ключ из переменной окружения
#     auth_data: str = API_KEY_GIGACHAT

#     @property
#     def metadata(self) -> LLMMetadata:
#         return LLMMetadata(
#             context_window=self.context_window,
#             num_output=self.num_output,
#             is_chat_model=True,
#             model_name=self.model_name,
#         )

#     @llm_completion_callback()
#     def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
#         if not self.auth_data:
#             raise ValueError("❌ Ошибка: Переменная GIGACHAT_AUTH_KEY не задана!")
            
#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.0,
#             "stream": False # Здесь обычный запрос
#         }

#         with GigaChatSDK(credentials=self.auth_data, verify_ssl_certs=False) as giga:
#             response = giga.chat(payload)
#             text = response.choices[0].message.content
            
#         return CompletionResponse(text=text)

#     @llm_completion_callback()
#     def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
#         """🔥 НАСТОЯЩИЙ СТРИМИНГ ДЛЯ СБЕРА: отдаем буквы по очереди во фронтенд"""
#         if not self.auth_data:
#             raise ValueError("❌ Ошибка: Переменная GIGACHAT_AUTH_KEY не задана!")

#         payload = {
#             "model": self.model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.0,
#             "stream": True # 🔥 ПРИКАЗЫВАЕМ СБЕРУ СТРИМИТЬ ОТВЕТ
#         }

#         def gen():
#             with GigaChatSDK(credentials=self.auth_data, verify_ssl_certs=False) as giga:
#                 # Используем метод обсчета потока от Сбера
#                 for chunk in giga.stream(payload):
#                     content = chunk.choices[0].delta.content
#                     if content:
#                         # Отдаем каждый кусочек текста в LlamaIndex по мере прилета из облака
#                         yield CompletionResponse(text=content, delta=content)
#         return gen()

# Settings.llm = LlamaGigaChat()


# ===== FIX: отключаем reasoning (think=False) на верхнем уровне /api/chat =====
# В установленной llama-index-llms-ollama==0.1.3 (контейнер) нет поля `thinking`,
# а `additional_kwargs` уходят внутрь `options`, где think игнорируется сервером.
# Поле `thinking` появилось в более новых версиях (host-venv 0.9.1).
# Минимальный подкласс: только для фактического streaming-метода добавляем
# top-level think=False, делегируя остальное в родителя (все options сохраняются).

class NoThinkOllama(Ollama):
    def stream_chat(self, messages, **kwargs):
        # В 0.1.3 payload = {...; "options": self._model_kwargs; "stream": True; **kwargs},
        # поэтому think=False через kwargs уходит НА ВЕРХНИЙ уровень, а не в options.
        kwargs["think"] = False
        return super().stream_chat(messages, **kwargs)

Settings.llm = NoThinkOllama(
    model="yagpt5_fns:latest",
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    temperature=0.0,
    context_window=8144,
    additional_kwargs={
        "keep_alive": -1,
        "num_predict": 512,
        "seed": 42,
        "num_ctx": 8144,
        "repeat_penalty": 1.05,
    },
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

# источники собираются в app.rag.sources.collect_sources


# Регулярное выражение для парсинга ID чанка: документ_статья_часть
_ARTICLE_CHUNK_ID_RE = re.compile(r'^(.+)_p(\d+)$')


# Служебные metadata-поля: используются только внутри retrieval/ranking,
# НИКОГДА не должны попадать ни в embedding, ни в LLM-контекст.
# Единый источник истины (module-level) для _init_bm25 и _sync_query.
_META_ONLY_KEYS = [
    "document_id", "point", "subpoint", "part", "total_parts",
    "subjects", "categories", "references", "keywords", "context_flat",
    "_em_input",
]



class RerankedEngine:

    # =========================================================
    # CONFIG
    # =========================================================
    VECTOR_WEIGHT = 0.65
    BM25_WEIGHT = 0.45

    BM25_TOP_K = 30
    RERANK_TOP_K = 10

    NEGATIVE_PATTERNS = [
        "не относится",
        "не является",
        "кроме",
        "не подлежит",
        "не подлежат",
        "не включается",
        "не включаются",
        "не входит",
        "не входят",
        "исключением",
        "за исключением",
    ]

    # Паттерны запросов на перечень/полноту (structure-aware reconstruction)
    _ENUMERATION_QUERY_PATTERNS = [
        "перечисли",
        "перечисли все",
        "какие виды",
        "какие бывают",
        "назови все",
        "укажи все",
        "полный перечень",
    ]

    QUERY_REPLACEMENTS = {
        "госслужащему": "государственному гражданскому служащему",
        "госслужащий": "гражданский служащий",
        "госслужба": "гражданская служба",
        "госслужащих": "государственных гражданских служащих",
        "на госслужбе": "на гражданской службе",
        "иноагент": "иностранный агент",
        "инагент": "иностранный агент",
        "работать": "проходить гражданскую службу",
        "увольнение": "прекращение служебного контракта",
        "начальник": "представитель нанимателя",
        "зарплата": "денежное содержание",
        "взятка": "коррупционное правонарушение",
        "отпуск": "ежегодный оплачиваемый отпуск",
        "коррупция": "коррупционное правонарушение",
        "коррупционный": "коррупционное правонарушение",
        "цкп": "цифровая кадровая платформа",
        "ё": "е"
    }

    BASE_ENTITIES = [
        "конкурс",
        "документы",
        "комиссия",
        "госслужащий",
        "отпуск",
        "контракт",
        "фнс",
        "коррупция",
        "служебная проверка",
    ]

    # =========================================================
    # INIT
    # =========================================================
    def __init__(
        self,
        index: Any,
        qa_prompt: Any,
        initial_top_k: int = 30,
        final_top_k: int = 5,
    ):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.qa_prompt = qa_prompt
        self.final_top_k = final_top_k
        self.stemmer = SnowballStemmer("russian")
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(exist_ok=True)
        self.cache_path = Path("bm25_cache.pkl")
        self.bm25 = None
        self.all_nodes = []
        self.node_map = {}
        # 🔥 КЕШ ТОКЕНОВ
        self.node_tokens_cache = {}
        self.reranker = None
        self.known_entities = set()
        logger.info("🛠 Init synthesizers...")


        self.compact_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=ResponseMode.COMPACT,
            use_async=False,
        )
        self.tree_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=ResponseMode.TREE_SUMMARIZE,
            use_async=False,
        )
        self.refine_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt,
            streaming=True,
            response_mode=ResponseMode.REFINE,
            use_async=False,
        )

        # =====================================================
        # RERANKER
        # =====================================================
        try:
            reranker_path = Path("/app/reranker")
            if reranker_path.exists():
                self.reranker = SentenceTransformerRerank(
                    model=str(reranker_path),
                    top_n=self.RERANK_TOP_K,
                )
                logger.info("✅ Reranker READY")
        except Exception as e:
            logger.error(f"⚠️ Reranker error: {e}")

        # =====================================================
        # BM25 + NODES
        # =====================================================
        try:
            self._init_bm25(index)
            self._load_graph_entities()
        except Exception as e:
            logger.error(f"❌ Init error: {e}", exc_info=True)

    # =========================================================
    # BM25 INIT
    # =========================================================
    def _init_bm25(self, index):
        loaded_from_cache = False

        if self.cache_path.exists():
            try:
                logger.info("📂 Loading BM25 cache...")
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    self.all_nodes = cache_data["nodes"]
                    self.bm25 = cache_data["bm25"]
                    loaded_from_cache = True
            except Exception as e:
                logger.warning(f"⚠️ BM25 cache invalid, rebuilding... {e}")
                logger.debug(traceback.format_exc())

        if not loaded_from_cache:
            logger.info("📡 Loading Qdrant nodes...")

            q_client = index.vector_store.client
            coll_name = index.vector_store.collection_name
            all_points = []
            next_page_offset = None

            while True:
                points, next_page_offset = q_client.scroll(
                    collection_name=coll_name,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                )
                all_points.extend(points)
                if next_page_offset is None:
                    break

            self.all_nodes = []

            for p in all_points:
                payload = p.payload or {}
                node_id = str(payload.get("id") or p.id)
                raw_content = payload.get("_node_content", "")
                node_text = ""

                if isinstance(raw_content, str) and raw_content.startswith("{"):
                    try:
                        node_text = json.loads(raw_content).get("text", "")
                    except BaseException:
                        node_text = raw_content

                if not node_text:
                    node_text = str(payload.get("text", ""))

                meta = {
                    "id": node_id,
                    "title": payload.get("title", "Документ"),
                    "source_url": payload.get("source_url", "http://kremlin.ru"),
                    "local_img": payload.get("local_img", ""),
                }

                # Сохраняем """ в """се остальные payload-поля (document_id, point, part,
                # total_parts, categories, subjects и т.д.) для internal-использования
                # в ranking/boost — но не для LLM.
                _INTERNAL_PAYLOAD_SKIP = {"_node_content", "_node_type", "doc_id", "ref_doc_id"}
                for pk in payload:
                    if pk not in meta and pk not in _INTERNAL_PAYLOAD_SKIP:
                        meta[pk] = payload[pk]

                # Поля, которые используются только внутри retrieval/ranking,
                # никогда не должны попадать ни в embedding, ни в LLM-контекст.
                # Единый список — _META_ONLY_KEYS (module-level константа).

                node = TextNode(
                    text=node_text,
                    id_=node_id,
                    metadata=meta,
                    excluded_embed_metadata_keys=["id", "source_url", "local_img", *_META_ONLY_KEYS],
                    excluded_llm_metadata_keys=[
                        "id", "source_url", "local_img", "graph_structure", "text", *_META_ONLY_KEYS
                    ],
                )
                node.metadata_template = "{key}: {value}"
                node.text_template = "РАЗДЕЛ: {metadata_str}\nТЕКСТ:\n{content}"
                self.all_nodes.append(node)

            if self.all_nodes:
                tokenized_corpus = []
                for node in self.all_nodes:
                    content = node.get_content(metadata_mode=MetadataMode.LLM)
                    tokens = self._tokenize(content)
                    tokenized_corpus.append(tokens)
                    # 🔥 КЕШ ТОКЕНОВ
                    self.node_tokens_cache[node.node_id] = set(tokens)

                self.bm25 = BM25Okapi(tokenized_corpus)

                with open(self.cache_path, "wb") as f:
                    pickle.dump(
                        {"nodes": self.all_nodes, "bm25": self.bm25},
                        f,
                    )

                logger.info("✅ BM25 cached")

        self.node_map = {n.node_id: n for n in self.all_nodes}

    # =========================================================
    # GRAPH ENTITIES
    # =========================================================
    def _load_graph_entities(self):
        try:
            if not os.path.exists("graph_global.json"):
                logger.warning("⚠️ graph_global.json not found")
                return

            with open("graph_global.json", "r", encoding="utf-8") as f:
                graph = json.load(f)

            for ent in graph.get("entities", []):
                name = ent.get("name", "").lower().strip()
                if len(name) >= 3:
                    self.known_entities.add(name)

            logger.info(f"✅ Graph entities: {len(self.known_entities)}")

        except Exception as e:
            logger.error(f"❌ Graph entity load error: {e}", exc_info=True)

    # =========================================================
    # NORMALIZE
    # =========================================================
    def _normalize_query(self, text: str) -> str:
        normalized = text.lower()
        for k in sorted(self.QUERY_REPLACEMENTS, key=len, reverse=True):
            normalized = normalized.replace(k, self.QUERY_REPLACEMENTS[k])
        return normalized

    # =========================================================
    # TOKENIZE
    # =========================================================
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        clean = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text.lower())
        tokens = clean.split()
        result = []

        for w in tokens:
            if w == "не":
                result.append(w)
            elif len(w) >= 2:
                result.append(self.stemmer.stem(w))

        return result

    # =========================================================
    # ENTITY EXTRACT
    # =========================================================
    def _extract_query_entities(self, query: str):
        query_tokens = set(self._tokenize(query.lower()))
        entities = []

        for ent in self.known_entities:
            ent_tokens = set(self._tokenize(ent))
            if ent_tokens & query_tokens:
                entities.append(ent)

        for ent in self.BASE_ENTITIES:
            ent_tokens = set(self._tokenize(ent))
            if ent_tokens & query_tokens and ent not in entities:
                entities.append(ent)

        return entities

    # =========================================================
    # RRF
    # =========================================================
    def _reciprocal_rank_fusion(self, vector_nodes, bm25_scores, k=30):
        scores = {}

        for rank, node in enumerate(vector_nodes):
            nid = str(node.node.metadata.get("id") or node.node.node_id)
            scores[nid] = scores.get(nid, 0) + self.VECTOR_WEIGHT / (k + rank + 1)

        bm25_indices = np.argsort(bm25_scores)[::-1][: self.BM25_TOP_K]

        for rank, idx in enumerate(bm25_indices):
            if bm25_scores[idx] <= 0:
                continue
            nid = self.all_nodes[idx].node_id
            scores[nid] = scores.get(nid, 0) + self.BM25_WEIGHT / (k + rank + 1)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            NodeWithScore(node=self.node_map[nid], score=scores[nid])
            for nid in sorted_ids
            if nid in self.node_map
        ]

    # =========================================================
    # RESPONSE MODE
    # =========================================================
    def _select_response_mode(self, query_text: str):
        q = query_text.lower()

        # --- 1. КОНТУР ГЛОБАЛЬНОЙ АНАЛИТИКИ И СУММАРИЗАЦИИ (ВРУБАЕМ TREE!) ---
        if any(p in q for p in ["сравни", "чем отличается", "разница", "обобщи",
                                "обзор", "анализ", "вывод", "кратко", "синтезируй",
                                "сформулируй"]):
            logger.info("🌲 GLOBAL -> TREE_SUMMARIZE")
            return self.tree_synthesizer

        # --- 2. КОНТУР ОТРИЦАНИЙ ---
        if any(p in q for p in self.NEGATIVE_PATTERNS):
            logger.info("⚡ COMPACT -> NEGATIVE")
            return self.compact_synthesizer

        # --- 3. КОНТУР БИОГРАФИЙ И ГРАФИКОВ ---
        if any(p in q for p in ["кто такой", "кто такая", "биография",
                                "руководитель", "график", "диаграмма"]):
            logger.info("👤 COMPACT -> BIO")
            return self.compact_synthesizer

        # --- 4. КОНТУР ТОЧНЫХ ОПЕРАТИВНЫХ ФАКТОВ ---
        if any(p in q for p in ["сколько", "какой срок", "когда",
                                "предусмотрено ли", "можно ли", "каким"]):
            logger.info("🎯 COMPACT -> FACT")
            return self.compact_synthesizer

        # --- 5. ДЕФОЛТНЫЙ КОНТУР ---
        logger.info("👤 COMPACT -> DEFAULT")
        return self.compact_synthesizer

    # =========================================================
    # QUERY
    # =========================================================
    def _dump_debug_info(self, query: str, norm_query: str, nodes: list):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.debug_dir / f"query_{ts}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"QUERY: {query}\nNORM: {norm_query}\n\n")
                for i, n in enumerate(nodes):
                    llm_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
                    f.write(f"\n[CHUNK {i + 1}] ID: {n.node.id_} | SCORE: {n.score:.4f}\n{'-' * 30}\n{llm_content}\n")
        except Exception as e:
            logger.error(f"❌ Debug Error: {e}", exc_info=True)

    def _sync_query(self, query_text: str):
        from llama_index.core.schema import MetadataMode, QueryBundle, NodeWithScore

        norm_query = self._normalize_query(query_text)

        # 1. VECTOR SEARCH
        vector_nodes = self.retriever.retrieve(norm_query)

        # 2. BM25 + HYBRID
        if self.bm25 and vector_nodes:
            bm25_scores = self.bm25.get_scores(self._tokenize(norm_query))
            combined_nodes = self._reciprocal_rank_fusion(vector_nodes, bm25_scores)
        else:
            combined_nodes = vector_nodes

        # ДЕБАГ HYBRID
        logger.info(f"\n{'=' * 20} HYBRID TOP-10 {'=' * 20}")
        for i, n in enumerate(combined_nodes[:10]):
            logger.info(f"Rank {i + 1}: [{n.score:.4f}] ID: {n.node.id_}")
        logger.info("=" * 55 + "\n")

        # 4. RERANK & STRICT SCORE FILTERING
        if self.reranker and combined_nodes:
            reranked_nodes = self.reranker.postprocess_nodes(
                combined_nodes[:10],
                query_bundle=QueryBundle(query_text),
            )
            top_5_reranked = reranked_nodes[:5]
            SCORE_THRESHOLD = 0.05
            # final_nodes = [node for node in top_5_reranked if node.score >= SCORE_THRESHOLD]
            final_nodes = top_5_reranked
            logger.info(
                f"🛡️ [BGE RERANK FILTER]: Из 5 переранжированных чанков "
                f"проверку по порогу >= {SCORE_THRESHOLD} прошли строго {len(final_nodes)}."
            )
        else:
            final_nodes = combined_nodes[:self.final_top_k]

        # ДЕБАГ РЕРАНК
        logger.info(f"\n{'=' * 20} RERANKED TOP-5 {'=' * 20}")
        for i, n in enumerate(final_nodes[:5]):
            logger.info(f"Rank {i + 1}: [{n.score:.4f}] ID: {n.node.id_}")
        logger.info("=" * 55 + "\n")

        # ИНИЦИАЛИЗАЦИЯ СИНТЕЗАТОРА И КОНТЕКСТА
        synthesizer = self._select_response_mode(query_text)
        final_chunks = final_nodes[:self.final_top_k]
        logger.info(f"🧬 Final chunks allowed for LLM context: {len(final_chunks)}")


        # Явно включаем title/source_url из metadata в LLM-контекст (вместо чистого {content})
        for nws in final_chunks:
            nws.node.text_template = "📄 {metadata_str}\n{content}"
            nws.node.metadata_template = "{key}: {value}"
            nws.node.excluded_llm_metadata_keys = [
                "id", "source_url", "local_img", "graph_structure", "text",
                *_META_ONLY_KEYS,
            ]

        # 🔥 УМНЫЙ ДЕБАГ: Пишем файлы строго если флаг включен в .env
        if os.getenv("DEBUG_MODE", "False").lower() == "true":
            import asyncio
            asyncio.run(asyncio.to_thread(
                self._dump_debug_info, query_text, norm_query, final_chunks
            ))
        else:
            logger.info("ℹ️ Debug dump skipped (Production mode)")

        # Флаг для безопасной заглушки на случай пустого контекста
        is_empty_context = not final_chunks

        # 🔥 МГНОВЕННЫЙ ЛОКАЛЬНЫЙ РАСЧЕТ ТОКЕНОВ
        try:
            prompt_str = str(self.qa_prompt) + "\n"
            full_input_text = prompt_str
            for chunk in final_chunks:
                full_input_text += chunk.node.get_content(
                    metadata_mode=MetadataMode.LLM
                ) + "\n"
            full_input_text += query_text

            exact_prompt_tokens = (
                max(1, int(len(full_input_text) / 4))
                if not is_empty_context
                else 0
            )
        except Exception as e:
            exact_prompt_tokens = f"Ошибка подсчета: {e}"

        # Запускаем оригинальный синтез стрима (с защитой от nodes=[])
                # ===== DIAGNOSTIC DUMP v2 =====
        _diag_ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        _diag_id = str(uuid.uuid4())[:8]
        _diag_path = "/tmp/rag_diag_" + _diag_ts + "_" + _diag_id
        try:
            with open(_diag_path + "_context.txt", "w", encoding="utf-8") as _f:
                _f.write("QUERY_TEXT: %s\n" % query_text)
                _f.write("NORM_QUERY: %s\n" % norm_query)
                _f.write("QUESTION: %s\n" % query_text)
                _f.write("NUM_CHUNKS: %d\n" % len(final_chunks))
                for _i, _nws in enumerate(final_chunks):
                    _f.write("\n--- CHUNK %d ---\n" % (_i + 1))
                    _f.write("  ID: %s\n" % _nws.node.node_id)
                    _f.write("  SCORE: %s\n" % str(_nws.score))
                    _f.write("  CONTENT: %s\n" % _nws.node.get_content(metadata_mode=MetadataMode.LLM))
        except Exception as _e:
            import logging
            logging.getLogger().error("DIAG dump error: %s" % str(_e))
        # ============================




        if not is_empty_context:
            streaming_response = synthesizer.synthesize(
                query=query_text,
                nodes=final_chunks,
            )
        else:
            fallback_msg = ("[ДАННЫЕ_НЕ_НАЙДЕНЫ: "
                            "В предоставленных документах ФНС информация отсутствует]")
            streaming_response = type(
                '_', (object,),
                {'response_gen': iter([fallback_msg])}
            )()

        # Перехватываем выходные токены через наш логгер
        original_gen = streaming_response.response_gen

        def logging_token_generator():
            tokens = []
            for token_obj in original_gen:
                if hasattr(token_obj, "delta"):
                    tokens.append(str(token_obj.delta))
                else:
                    tokens.append(str(token_obj))
                yield token_obj

            generated_text = "".join(tokens)
            token_count = (
                max(1, int(len(generated_text) / 4))
                if not is_empty_context
                else 0
            )

            logger.info("═" * 50)
            logger.info("📊 ТОЧНЫЙ АУДИТ ТОКЕНОВ ДЛЯ ФНС (ЛОКАЛЬНЫЙ РАСЧЕТ):")
            logger.info(
                f"📥 На вход улетело (Промпт + Чанки + Вопрос): "
                f"~{exact_prompt_tokens} токенов"
            )
            logger.info(
                f"📤 На выход сгенерировано моделью: ~{token_count} токенов"
            )
            logger.info("═" * 50)

        streaming_response.response_gen = logging_token_generator()

        return streaming_response


# ========== ИНИЦИАЛИЗАЦИЯ ==========
logger.info("🧠 Загрузка векторного индекса...")

# 1. Создаем клиент ЯВНО (параметры из .env, дефолты для Docker Compose)
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant_db")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False)

# 2. Передаем его в стор
vector_store = QdrantVectorStore(
    collection_name=os.getenv("QDRANT_COLLECTION", "fns_collection"),
    client=client
)

# 3. Собираем индекс
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Инициализируем твой движок
query_engine = RerankedEngine(
    index=index,
    qa_prompt=qa_prompt,
    initial_top_k=30,
    final_top_k=5,
)
logger.info("✅ Query engine ТЕПЕРЬ РЕАЛЬНО НА QDRANT!")


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
# Обёртка над общим модулем источников (app.rag.sources).
def _collect_sources(nodes: list, max_sources: int = 3) -> list:
    return collect_sources(nodes, max_sources=max_sources)



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

        # 1. Проверяем намерение пользователя через изолированный chart_engine
        is_chart_mode = chart_engine.is_chart_request(query_text)

        # 2. В QDRANT ШЛЕМ СТРОГО ЧИСТЫЙ ВОПРОС (в потоке, чтобы не вешать Event Loop)
        response = await asyncio.to_thread(query_engine._sync_query, query_text)

        if response is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."},
                             ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        nodes = response.source_nodes if hasattr(response, "source_nodes") else []
        has_real_context = bool(nodes)

        # SOURCE INPUT DIAGNOSTIC
        logger.info("═" * 40)
        logger.info("📋 SOURCE INPUT")
        for _i, _nws in enumerate(nodes):
            _meta = getattr(_nws, "node", _nws)
            _meta = getattr(_meta, "metadata", {}) if hasattr(_meta, "metadata") else {}
            logger.info(
                f"  Rank {_i+1} | node_id={getattr(getattr(_nws, 'node', _nws), 'node_id', '?')} "
                f"| score={getattr(_nws, 'score', '?'):.4f} "
                f"| source_url={_meta.get('source_url', '?')} "
                f"| title={_meta.get('title', '?')[:60]}"
            )
        logger.info("═" * 40)

        sources = _collect_sources(nodes)

        # SOURCE OUTPUT DIAGNOSTIC
        logger.info("═" * 40)
        logger.info("📋 SOURCE OUTPUT")
        for _i, _src in enumerate(sources):
            logger.info(
                f"  Rank {_i+1} | url={_src.get('url', '?')} "
                f"| title={_src.get('title', '?')[:60]} "
                f"| score={_src.get('score', '?')}"
            )
        logger.info("═" * 40)

        logger.info(f"🧩 Источников для фронта: {len(sources)}")
        local_img = nodes[0].node.metadata.get('local_img', '') if nodes else ''

        # Отправляем метаданные и источники на фронтенд
        yield json.dumps({
            "type": "metadata",
            "sources": sources,
            "has_answer": has_real_context,
            "img": local_img,
        }, ensure_ascii=False) + "\n"

        if not has_real_context or not hasattr(response, "response_gen") or response.response_gen is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."},
                             ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        gen_start = time.time()
        tokens: List[str] = []

        # =========================================================
        # РАЗВЕТВЛЕНИЕ КОНТУРОВ: ГРАФИК VS СТАНДАРТНЫЙ ТЕКСТ
        # =========================================================
        if is_chart_mode:
            logger.info("🎯 [API]: Включаем изолированный Pydantic-контур генерации графика.")

            if not has_real_context:
                logger.warning("⚠️ [CHART]: Нет данных для графика, переключаюсь на текстовый ответ.")
                for token in response.response_gen:
                    tokens.append(token)
                    t = str(token)
                    yield json.dumps({"type": "text", "content": t}, ensure_ascii=False) + "\n"
                full_response_text = "".join(tokens)

            else:
                rag_context = "\n\n".join([node.node.get_content() for node in nodes])

                config = chart_engine.process_llm_payload(
                    query=query_text,
                    rag_context=rag_context,
                    model_name="yagpt5_fns:latest"
                )
                payload = config["payload"]

                try:
                    import httpx
                    async with httpx.AsyncClient() as http_client:
                        ollama_response = await http_client.post(
                            chart_engine.ollama_url, json=payload, timeout=300.0
                        )
                        ollama_response.raise_for_status()
                        raw_json_text = ollama_response.json()["message"]["content"]
                except Exception as chart_err:
                    logger.error(f"❌ [CHART] Ошибка вызова Ollama: {chart_err}", exc_info=True)
                    for token in response.response_gen:
                        tokens.append(token)
                        t = str(token)
                        yield json.dumps({"type": "text", "content": t}, ensure_ascii=False) + "\n"
                    full_response_text = "".join(tokens)
                    is_chart_mode = False
                    yield json.dumps(
                        {"type": "metadata", "note": "График не построен, показан текстовый ответ"},
                        ensure_ascii=False,
                    ) + "\n"

                tokens = list(raw_json_text)
                parsed_chart_node = chart_engine.validate_and_parse(raw_json_text)

                if parsed_chart_node["type"] == "chart_error":
                    logger.warning(
                        f"⚠️ [CHART] Валидация не прошла: {parsed_chart_node['message']}. "
                        "Отправляю ошибку на фронт."
                    )
                    yield json.dumps(parsed_chart_node, ensure_ascii=False) + "\n"
                    full_response_text = ""
                else:
                    yield json.dumps(parsed_chart_node, ensure_ascii=False) + "\n"
                    logger.info("📊 График успешно отвалидирован и отправлен на фронт.")

        else:
            # Сценарий Б: Стандартный стриминг текстовых токенов
            for token in response.response_gen:
                tokens.append(token)
                t = str(token)
                yield json.dumps({"type": "text", "content": t}, ensure_ascii=False) + "\n"
            full_response_text = "".join(tokens)

        # Логируем скорость работы
        gen_time = time.time() - gen_start
        token_count = len(tokens)
        if gen_time > 0 and not is_chart_mode:
            logger.info(
                f"💬 {token_count} токенов за {gen_time:.2f} сек "
                f"({token_count / gen_time:.1f} ток/сек)"
            )

        # =========================================================
        # ПОДБОР ФОТОГРАФИЙ
        # =========================================================
        if not is_chart_mode:
            resp_lower = full_response_text.lower()
            is_empty = bool(_EMPTY_RESPONSE_RE.search(resp_lower))
            is_table = "|---" in resp_lower or "| :---" in resp_lower or resp_lower.count("|") > 10

            if not is_empty and not is_table:
                final_photo = _find_photo(resp_lower, nodes)
                if final_photo and "ии-помощник" not in resp_lower:
                    encoded = urllib.parse.quote(final_photo)
                    yield json.dumps(
                        {"type": "text", "content": f"\n\n![photo](/images/{encoded})"},
                        ensure_ascii=False,
                    ) + "\n"
                    logger.info(f"📸 Добавлено фото: {final_photo}")

        # Закрываем стрим
        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
        logger.info(f"⏱️ Итого: {time.time() - start_time:.2f} сек")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        yield json.dumps({"type": "error", "content": f"Ошибка сервера: {e}"},
                         ensure_ascii=False) + "\n"
        yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"


# ========== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ (без стриминга) ==========
async def get_ai_response_full(query_text: str) -> dict:
    try:
        logger.info(f"📝 Синхронный запрос: '{query_text[:100]}...'")
        response = await query_engine.aquery(query_text)

        if not response or not hasattr(response, "source_nodes"):
            return {
                "answer": "БАЗА_ПУСТА: Информация не найдена.",
                "sources": [],
                "image": None,
            }

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
        return {
            "answer": "В моих регламентах про это ни слова, бро.",
            "sources": [],
            "image": None,
        }


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