"""Builds the grounded prompt and calls the Groq API."""
from __future__ import annotations

import logging
import os
from groq import Groq

from app.services.retrieval import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
   "You are a document assistant. Answer ONLY using the numbered context passages "
    "given to you. You must not use outside knowledge.\n"
    "Rules:\n"
    "1. Every factual sentence must be followed by its citation marker, e.g. [1] or [2].\n"
    "2. If the user asks for a summary, overview, or main topics, summarize and synthesize the key points from the provided context passages and cite the sources used.\n"
    "3. If the context does not contain relevant information to answer or summarize, reply exactly: "
    "\"I could not find this in the provided documents.\"\n"
    "4. Keep specific answers concrete and quote exact numbers and settings when present."
)

USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer using only the passages above, with [n] citations."""


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.label})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


class Generator:
    def __init__(self, model: str = "qwen/qwen3.8-27b", api_key: str | None = None, **kwargs):
        self.model = "qwen/qwen3.8-27b"
        key = api_key or os.getenv("GROQ_API_KEY") or "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"
        
        if key:
            self.client = Groq(api_key=key)
        else:
            self.client = None

    def health(self) -> bool:
        """True if Groq API client is initialized."""
        return self.client is not None

    def answer(self, question: str, chunks: list[Chunk], history: list[dict] = None) -> str:
     if not chunks:
        return "I could not find this in the provided documents."

     if not hasattr(self, "client") or self.client is None:
        return "Groq Client is not initialized. Please check your GROQ_API_KEY."

    # 1. تهيئة مصفوفة الرسائل بالـ SYSTEM_PROMPT
     messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 2. إرفاق آخر 3 دورات حوارية من الـ History إن وجدت
     if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 3. إرفاق السياق والسؤال الحالي
     user_content = USER_TEMPLATE.format(
        context=build_context(chunks), question=question
     )
     messages.append({"role": "user", "content": user_content})

     try:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
     except Exception as exc:
        logger.error("Groq API Call Failed: %s", exc)
        return f"Groq Error: {exc}"