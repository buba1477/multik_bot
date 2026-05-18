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

from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, models

from ollama import ChatResponse
from datetime import datetime
import pickle

from llama_index.core.schema import MetadataMode # <--- ДОБАВЬ MetadataMode


from qdrant_client import QdrantClient # СТРОГО ТАК
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

def patched_setitem(self, key, value):
    try:
        # Пытаемся записать нормально
        object.__setattr__(self, key, value)
    except Exception:
        # Если Pydantic орет — просто забиваем болт на это поле
        pass

# Подменяем метод записи во всей библиотеке на лету
ChatResponse.__setitem__ = patched_setitem


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
 
_QA_PROMPT_STR = """
Ты — система точного поиска по законодательству РФ.

Отвечай только на основе контекста.

Разрешается:
- делать краткий вывод, если он прямо следует из текста;
- использовать несколько связанных фрагментов контекста.

Запрещено:
- придумывать информацию;
- использовать знания вне контекста;
- делать собственные юридические выводы;
- подменять тему вопроса смежными нормами.

Если ответа в контексте нет — верни строго:
БАЗА_ПУСТА

Если вопрос содержит:
- график
- диаграмма
- круговая
- столбчатая
- гистограмма
- динамика

то:
- верни ТОЛЬКО блок [CHART_JSON]...[/CHART_JSON];
- JSON должен быть валидным;
- внутри только JSON;
- никаких пояснений вне блока.

Пример:
[CHART_JSON]{{"title":{{"text":"Название"}},"xAxis":{{"type":"category","data":["A","B"]}},"yAxis":{{"type":"value"}},"series":[{{"type":"bar","data":[1,2]}}]}}[/CHART_JSON]

------------------------
КОНТЕКСТ:
{context_str}
------------------------

ВОПРОС:
{query_str}

ОТВЕТ:
"""

qa_prompt = PromptTemplate(_QA_PROMPT_STR)


_FORCED_QUERY_SUFFIX = "\n\nВАЖНО: ОТВЕТЬ НА РУССКОМ ЯЗЫКЕ, ИСПОЛЬЗУЯ ТОЛЬКО БАЗУ ЗНАНИЙ."


# ========== ЭМБЕДДЕР ==========
logger.info(f"✨ Загрузка эмбеддера RoSBERTa на CPU: {MODEL_PATH}")

class SberRoSBERTaEmbedding(BaseEmbedding):
    _model: Any = PrivateAttr()

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"✨ Загрузка эмбеддера RoSBERTa на {device}")
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), 
            pooling_mode='cls'
        )
        self._model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

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

Settings.llm = Ollama(
    model="yagpt5_fns:latest",
    base_url=OLLAMA_HOST,
    request_timeout=300.0,  # Чуть поднял, так как реранкер + 8b на 1660ti могут занять время
    temperature=0.0,        # Жесткий ноль снаружи
    context_window=4096,    # Контекст для LlamaIndex снаружи
    
    # 🔥 ВСЕ параметры генерации для Ollama должны идти строго в options!
    options={
        "seed": 42,
        "num_ctx": 4096,     # Дублируем строго внутри опций самого движка Ollama
        "num_predict": 512,
        "repeat_penalty": 1.05,
    },
    
    # В additional_kwargs оставляем только системные параметры самого контейнера
    additional_kwargs={
        "keep_alive": "-1"   # Держим модель в памяти GPU вечно
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
    def __init__(self, index: Any, qa_prompt: Any, initial_top_k: int = 30, final_top_k: int = 3):
        self.retriever = index.as_retriever(similarity_top_k=initial_top_k)
        self.qa_prompt = qa_prompt
        self.final_top_k = final_top_k
        self.stemmer = SnowballStemmer("russian")
        
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(exist_ok=True)
        self.cache_path = Path("bm25_cache.pkl")
        
   
        logger.info("🛠 Инициализация synthesizers...")
        self.compact_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt, streaming=True, 
            response_mode=ResponseMode.COMPACT, use_async=False
        )
        self.tree_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt, streaming=True, 
            response_mode=ResponseMode.TREE_SUMMARIZE, use_async=False
        )
        self.refine_synthesizer = get_response_synthesizer(
            text_qa_template=self.qa_prompt, streaming=True, 
            response_mode=ResponseMode.REFINE, use_async=False
        )

        self.bm25 = None
        self.all_nodes = []
        self.node_map = {}
        self.reranker = None
        
        self.known_entities = set()
        # 1. Реранкер
        try:
            reranker_path = Path("/app/reranker")
            if reranker_path.exists():
                self.reranker = SentenceTransformerRerank(model=str(reranker_path), top_n=10)
                logger.info("✅ Реранкер READY")
        except Exception as e:
            logger.error(f"⚠️ Реранкер Error: {e}")

        # 2. Узлы и BM25
        try:
            loaded_from_cache = False
            if self.cache_path.exists():
                try:
                    logger.info("📂 Загрузка BM25 из кэша...")
                    with open(self.cache_path, "rb") as f:
                        cache_data = pickle.load(f)
                        self.all_nodes = cache_data['nodes']
                        self.bm25 = cache_data['bm25']
                        loaded_from_cache = True
                except Exception:
                    logger.warning("⚠️ Кэш несовместим, пересобираю...")

            if not loaded_from_cache:
                logger.info("📡 Потоковая выгрузка из Qdrant (Scroll)...")
                q_client = index.vector_store.client
                coll_name = index.vector_store.collection_name
                
                all_points = []
                next_page_offset = None
                while True:
                    points, next_page_offset = q_client.scroll(
                        collection_name=coll_name, limit=1000, offset=next_page_offset, with_payload=True
                    )
                    all_points.extend(points)
                    if next_page_offset is None: break
                
                self.all_nodes = []
                for p in all_points:
                    payload = p.payload or {}
                    node_id = str(payload.get("id") or p.id)
                    raw_content = payload.get("_node_content", "")
                    node_text = ""
                    if isinstance(raw_content, str) and raw_content.startswith('{'):
                        try: node_text = json.loads(raw_content).get("text", "")
                        except: node_text = raw_content
                    if not node_text: node_text = str(payload.get("text", ""))

                    meta = {
                        "id": node_id,
                        "title": payload.get("title", "Документ"),
                        "source_url": payload.get("source_url", "http://kremlin.ru"),
                        "local_img": payload.get("local_img", "")
                    }
                    
                    node = TextNode(
                        text=node_text, 
                        id_=node_id, 
                        metadata=meta,
                        excluded_embed_metadata_keys=["id", "source_url", "local_img"],
                        excluded_llm_metadata_keys=["id", "source_url", "local_img", "graph_structure"]
                    )
                    node.metadata_template = "{key}: {value}"
                    node.text_template = "РАЗДЕЛ: {metadata_str}\nТЕКСТ:\n{content}"
                    self.all_nodes.append(node)

                if self.all_nodes:
                    tokenized_corpus = [self._tokenize(n.get_content(metadata_mode=MetadataMode.LLM)) for n in self.all_nodes]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    with open(self.cache_path, "wb") as f:
                        pickle.dump({'nodes': self.all_nodes, 'bm25': self.bm25}, f)
                    logger.info("✅ BM25 закэширован")

            if self.all_nodes:
                self.node_map = {n.node_id: n for n in self.all_nodes}
                # ========================================
                # LOAD GRAPH ENTITIES
                # ========================================

                try:

                    if os.path.exists("graph_global.json"):

                        with open("graph_global.json", "r", encoding="utf-8") as f:

                            graph = json.load(f)

                            for ent in graph.get("entities", []):

                                name = ent.get("name", "").lower().strip()

                                if len(name) >= 3:
                                    self.known_entities.add(name)

                        logger.info(
                            f"✅ Graph entities loaded: {len(self.known_entities)}"
                        )

                    else:

                        logger.warning(
                            "⚠️ graph_global.json not found"
                        )

                except Exception as e:

                    logger.error(
                        f"❌ Graph entity load error: {e}"
    )
        except Exception as e:
            logger.error(f"❌ Ошибка init: {e}", exc_info=True)
    
    def _normalize_query(self, text: str) -> str:
        replacements = {
            # госслужба
            "госслужащему": "гражданскому служащему",
            "госслужащий": "гражданский служащий",
            "госслужба": "гражданская служба",
            "на госслужбе": "на гражданской службе",

            # иноагенты
            "иноагент": "иностранный агент",
            "инагент": "иностранный агент",

            # работа / увольнение
            "работать": "проходить гражданскую службу",
            "увольнение": "прекращение служебного контракта",

            # бытовые слова
            "начальник": "представитель нанимателя",
            "зарплата": "денежное содержание",
            "взятка": "коррупционное правонарушение",
            "отпуск": "ежегодный оплачиваемый отпуск",
            "коррупция": "коррупционное правонарушение",
            "коррупционный": "коррупционное правонарушение",
        }

        normalized = text.lower()

        # ВАЖНО:
        # длинные фразы сначала
        for k in sorted(replacements, key=len, reverse=True):
            normalized = normalized.replace(k, replacements[k])

        return normalized

    def _tokenize(self, text: str) -> List[str]:
        if not text: return []
        clean = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', text.lower())
        tokens = clean.split()
        result = []
        for w in tokens:
            if w == "не": result.append(w)
            elif len(w) >= 2: result.append(self.stemmer.stem(w))
        return result

    def _reciprocal_rank_fusion(self, vector_nodes, bm25_scores, k=30):
        scores = {}
        for rank, node in enumerate(vector_nodes):
            nid = str(node.node.metadata.get('id') or node.node.node_id)
            scores[nid] = scores.get(nid, 0) + 0.4 / (k + rank + 1)
        bm25_indices = np.argsort(bm25_scores)[::-1][:30]
        for rank, idx in enumerate(bm25_indices):
            if bm25_scores[idx] <= 0: continue
            nid = self.all_nodes[idx].node_id
            scores[nid] = scores.get(nid, 0) + 0.6 / (k + rank + 1)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [NodeWithScore(node=self.node_map[nid], score=scores[nid]) for nid in sorted_ids if nid in self.node_map]

    def _dump_debug_info(self, query: str, norm_query: str, nodes: list, final_prompt: str):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.debug_dir / f"query_{ts}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"ORIGINAL QUERY: {query}\nNORM SEARCH: {norm_query}\n\nPROMPT:\n{final_prompt}\n")
                f.write(f"{'='*60}\n")
                for i, n in enumerate(nodes):
                    llm_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
                    f.write(f"\n[CHUNK {i+1}] ID: {n.node.id_} | SCORE: {n.score:.4f}\n{'-'*30}\n{llm_content}\n")
        except Exception as e:
            logger.error(f"❌ Debug Error: {e}")

    # 🔥 ФУНКЦИЯ ВЫБОРА РЕЖИМА (ROUTER)
    def _select_response_mode(self, query_text: str):
        q = query_text.lower()

        # 1. NEGATIVE / MCQ
        negative_router_patterns = [
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
            "выберите",
            "верно ли",
            "что из перечисленного",
        ]

        if any(p in q for p in negative_router_patterns):
            logger.info("⚡ COMPACT -> NEGATIVE")
            return self.compact_synthesizer

        # 2. BIO / PERSON
        if any(p in q for p in [
            "кто такой",
            "кто такая",
            "биография",
            "родился",
            "руководитель",
            "заместитель",
            "график",
            "диаграмма",
            "круговая",
            "столбчатая",
            "гистограмма",
            "динамика"
        ]):
            logger.info("👤 COMPACT -> BIO")
            return self.compact_synthesizer

        # 3. FACT
        if any(p in q for p in [
            "сколько",
            "какой срок",
            "когда",
            "предусмотрено ли",
            "разрешается ли",
            "допускается ли",
            "можно ли"
        ]):
            logger.info("🎯 COMPACT -> FACT")
            return self.compact_synthesizer

        # 4. SUMMARY
        if any(p in q for p in [
            "расскажи",
            "объясни",
            "перечисли",
            "как работает",
            "какие бывают",
            "в каких случаях"
        ]):
            logger.info("🌲 TREE -> SUMMARY")
            return self.tree_synthesizer

        logger.info("🌲 TREE -> DEFAULT")
        return self.tree_synthesizer
    
    def _extract_query_entities(self, query: str):
    
        query = query.lower()

        entities = []

        base_entities = [
            "конкурс",
            "документы",
            "комиссия",
            "госслужащий",
            "отпуск",
            "контракт",
            "фнс",
            "коррупция",
            "служебная проверка"
        ]

        query_tokens = set(self._tokenize(query))

        for ent in self.known_entities:

            ent_tokens = set(self._tokenize(ent))

            if ent_tokens & query_tokens:
                entities.append(ent)

        for ent in base_entities:

            ent_tokens = set(self._tokenize(ent))

            if ent_tokens & query_tokens and ent not in entities:
                entities.append(ent)

        return entities

    def _sync_query(self, query_text: str):
        norm_query = self._normalize_query(query_text)
        logger.info(f"🔎 [QUERY]: {query_text} -> [SEARCH]: {norm_query}")
        
        # 1. Hybrid Retrieval
        vector_nodes = self.retriever.retrieve(norm_query)

        if self.bm25 and vector_nodes:
            bm25_scores = self.bm25.get_scores(self._tokenize(norm_query))
            combined_nodes = self._reciprocal_rank_fusion(vector_nodes, bm25_scores)

            query_entities = self._extract_query_entities(query_text)

            # кешируем токены entity один раз
            entity_token_cache = {
                ent: set(self._tokenize(ent))
                for ent in query_entities
            }

            for node in combined_nodes:

                text = node.node.text.lower()

                bonus = 0.0

                text_tokens = set(self._tokenize(text))

                for ent_tokens in entity_token_cache.values():

                    if ent_tokens & text_tokens:
                        bonus += 0.02

                bonus = min(bonus, 0.08)
                node.score *= (1.0 + bonus)

            combined_nodes = sorted(
                combined_nodes,
                key=lambda x: x.score,
                reverse=True
            )
        else:
            combined_nodes = vector_nodes

        # 🔥 ВЕРНУЛ ЛОГИ ГИБРИДА
        print(f"\n{'='*20} HYBRID TOP-10 {'='*20}")
        for i, n in enumerate(combined_nodes[:10]):
            print(f"Rank {i+1}: [{n.score:.4f}] ID: {n.node.id_}")
        print("="*55 + "\n")

        # 2. Reranking (Cross-Encoder)
        if self.reranker and combined_nodes:
            final_nodes = self.reranker.postprocess_nodes(
                combined_nodes[:10], 
                query_bundle=QueryBundle(query_text) # Используем оригинал
            )
            # 🔥 ВЕРНУЛ ЛОГИ РЕРАНКЕРА
            print(f"{'='*20} RERANKED TOP-5 {'='*20}")
            for i, n in enumerate(final_nodes[:5]):
                print(f"Rank {i+1}: [{n.score:.4f}] ID: {n.node.id_}")
            print("="*55 + "\n")
        else:
            final_nodes = combined_nodes[:self.final_top_k]

        # 3. Адаптивный фильтр

        filtered_nodes =  final_nodes
        
        
        # 4. Forced Query
       
        forced_query = (
            f"ВОПРОС:\n{query_text}\n\n"
            f"ВАЖНО:\n"
            f"- отвечай только по русски;\n"
            f"- не делать собственных юридических выводов.\n"
        )

        negative_patterns = [
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

        q_lower = query_text.lower()

        is_negative = any(
            p in q_lower
            for p in negative_patterns
        )

        if is_negative:
            forced_query += (
                "- вопрос содержит отрицание или исключение;\n"
                "- нужно найти вариант, который НЕ подтверждается контекстом;\n"
                "- остальные варианты могут подтверждаться законом;\n"
                "- отвечай только если в тексте есть прямое подтверждение;\n"
            )
        # 5. Дебаг и Синтез
        logger.info(f"🧬 Final chunks: {len(filtered_nodes[:self.final_top_k])}")
        total_chars = sum(len(c.node.get_content(metadata_mode=MetadataMode.LLM)) for c in filtered_nodes[:self.final_top_k])

        logger.info(f"TOTAL CHARS: {total_chars}")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

        # 🔥 ВЫБОР СИНТЕЗАТОРА
        synthesizer = self._select_response_mode(query_text)

        self._dump_debug_info(query_text, norm_query, filtered_nodes[:self.final_top_k], forced_query)
        
        return synthesizer.synthesize(
            query=forced_query, 
            nodes=filtered_nodes[:self.final_top_k]
        )

 
# ========== ИНИЦИАЛИЗАЦИЯ ==========
logger.info("🧠 Загрузка векторного индекса...")

# 1. Создаем клиент ЯВНО
# ВАЖНО: используем именно QdrantClient (он синхронный)
client = QdrantClient(host="qdrant_db", port=6333, prefer_grpc=False)

# 2. Передаем его в стор
vector_store = QdrantVectorStore(
    collection_name="fns_collection", 
    client=client
)

# 3. Собираем индекс
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Инициализируем твой движок
query_engine = RerankedEngine(
    index=index,
    qa_prompt=qa_prompt,
    initial_top_k=30,
    final_top_k=3,
)
logger.info("✅ Query engine ТЕПЕРЬ РЕАЛЬНО НА QDRANT!")

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

        response = query_engine._sync_query(query_text)

        if response is None:
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Информация не найдена."}, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"
            return

        nodes = response.source_nodes if hasattr(response, "source_nodes") else []
        has_real_context = bool(nodes)

        sources = _collect_sources(nodes)
        logger.info(f"🧩 Источников для фронта: {len(sources)}")
        local_img = nodes[0].node.metadata.get('local_img', '')

        yield json.dumps({
            "type": "metadata",
            "sources": sources,
            "has_answer": has_real_context,
            "img": local_img
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