"""Общие функции: чистка HTML, токенизация, метрика MAP@10.

Выносим сюда всё, что будет переиспользоваться между ноутбуками,
чтобы логика была в одном месте.
"""

from __future__ import annotations

import re

import snowballstemmer
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Чистка HTML
# ---------------------------------------------------------------------------

def clean_html(html: str) -> str:
    """Извлекает видимый текст из HTML статьи.

    Сохраняем содержимое таблиц, спойлеров и вкладок — там часто лежат
    сами инструкции. Удаляем только script/style.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Схлопываем пробелы/переносы
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Токенизация со стеммингом (для BM25)
# ---------------------------------------------------------------------------

_stemmer = snowballstemmer.stemmer("russian")
_token_re = re.compile(r"[а-яёa-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Нижний регистр -> токены (кириллица/латиница/цифры) -> стемминг."""
    tokens = _token_re.findall(text.lower().replace("ё", "е"))
    return _stemmer.stemWords(tokens)


# ---------------------------------------------------------------------------
# Метрика MAP@10 (по формуле из условия)
# ---------------------------------------------------------------------------

def ap_at_k(predicted: list[int], relevant: set[int], k: int = 10) -> float:
    """AP@k = (1 / min(|R|, k)) * sum_i [p_i in R] * Precision@i."""
    if not relevant:
        return 0.0
    predicted = predicted[:k]
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted, start=1):
        if p in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def map_at_k(preds: dict[int, list[int]], truth: dict[int, set[int]], k: int = 10) -> float:
    """Среднее AP@k по всем запросам из truth."""
    return sum(ap_at_k(preds[qid], rel, k) for qid, rel in truth.items()) / len(truth)


def parse_ground_truth(gt: str) -> set[int]:
    return {int(x) for x in str(gt).split()}


# ---------------------------------------------------------------------------
# Чанкование длинных статей (для эмбеддингов)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_words: int = 160, stride_words: int = 120) -> list[str]:
    """Режет текст на перекрывающиеся окна по словам.

    Перекрытие нужно, чтобы релевантный фрагмент не оказался разрезанным
    границей чанка. Для пустого/короткого текста возвращает один чанк.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text] if text else [""]
    chunks = []
    for start in range(0, len(words), stride_words):
        chunk = " ".join(words[start : start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks
