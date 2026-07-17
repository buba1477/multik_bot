"""
Единый модуль логирования для Multik RAG.
- Пишет в защищённый файл app_audit.log (ротация 10MB, 5 бэкапов)
- Дублирует в stdout (для Docker)
- Каждое сообщение содержит request_id для аудита
"""

import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar

# ContextVar для проброса request_id через цепочку вызовов
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="N/A")


def get_request_id() -> str:
    """Возвращает текущий request_id из контекста."""
    return _request_id_ctx.get()


def set_request_id(request_id: str | None = None) -> str:
    """Устанавливает request_id в контекст. Если не передан — генерирует новый."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:12]
    _request_id_ctx.set(request_id)
    return request_id


class RequestIdFilter(logging.Filter):
    """Фильтр, добавляющий request_id в каждую запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logger(name: str = "multik") -> logging.Logger:
    """Настраивает и возвращает логгер с двумя handler'ами.

    Args:
        name: Имя логгера (по умолчанию 'multik').

    Returns:
        Настроенный логгер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Предотвращаем дублирование handler'ов при повторных вызовах
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [request_id=%(request_id)s] "
        "[%(module)s:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f",
    )

    # 1. File handler — ротация, пишем в app_audit.log
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, "app_audit.log")

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIdFilter())

    # 2. Stream handler — дублируем в stdout (для Docker)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RequestIdFilter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# Создаём и экспортируем логгер по умолчанию
logger = setup_logger("multik")