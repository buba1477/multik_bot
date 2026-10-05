"""Работа с actual.pravo.gov.ru (API на порту 8000) — fallback для HTML-текста НПА.

API endpoints (реверс-инжиниринг JS):
1. GET /api/ebpi/redactions/?bpa=ebpi&t={"pnum":"<eo>"} → rdk
2. GET /api/ebpi/redtext?bpa=ebpi&t=<rdk>&ttl=2       → {"redtext":"<html>..."}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "http://actual.pravo.gov.ru:8000/api/ebpi"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


class ActualAPIError(RuntimeError):
    """Базовая ошибка actual.pravo.gov.ru."""


class DocumentTextNotFoundError(ActualAPIError):
    """Текст документа недоступен через actual.pravo.gov.ru."""


def _get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_redactions(eo_number: str) -> list[dict]:
    """
    Получить список редакций документа по eoNumber (pnum).

    GET /api/ebpi/redactions/?bpa=ebpi&t={"pnum":"<eo>","ttl":2}
    """
    t = json.dumps({"pnum": eo_number, "ttl": 2})
    url = f"{API_BASE}/redactions/?bpa=ebpi&t={urllib.parse.quote(t)}"
    data = _get_json(url)
    reds = data.get("redactions", [])
    if not reds:
        raise DocumentTextNotFoundError(f"Нет редакций для eoNumber={eo_number}")
    return reds


def get_redtext(rdk: int, ttl: int = 2) -> str:
    """
    Получить HTML-текст редакции по rdk (redaction id).

    GET /api/ebpi/redtext?bpa=ebpi&t=<rdk>&ttl=2
    """
    url = f"{API_BASE}/redtext?bpa=ebpi&t={rdk}&ttl={ttl}"
    data = _get_json(url)
    html = data.get("redtext", "")
    if not html or len(html) < 100:
        raise DocumentTextNotFoundError(f"Пустой текст для rdk={rdk}")
    return html


def download_html_from_actual(eo_number: str, dest: Path) -> int:
    """
    Скачать HTML текст документа из actual.pravo.gov.ru.

    Процесс:
      1. redactions — rdk первой актуальной редакции
      2. redtext — HTML текст

    Returns: размер сохранённого HTML в байтах.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    reds = get_redactions(eo_number)
    rdk = reds[0]["redid"]

    html = get_redtext(rdk)

    if "<!DOCTYPE html>" not in html and "<html" not in html:
        raise ActualAPIError(f"Ответ не HTML (eo={eo_number}, rdk={rdk})")

    dest.write_bytes(html.encode("utf-8"))
    size = dest.stat().st_size
    if size == 0:
        raise ActualAPIError(f"Пустой файл для eo={eo_number}")
    return size