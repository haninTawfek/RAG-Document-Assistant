"""Thin wrapper around the backend API. No Streamlit code lives here."""
from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Read from the environment - never hard-code the backend URL.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))

# مهلة مخصصة لرفع ومعالجة الملفات الكبيرة (10 دقائق)
UPLOAD_TIMEOUT = int(os.getenv("API_UPLOAD_TIMEOUT", "600"))


class APIError(Exception):
    """Raised when the backend is unreachable or returns an error."""


def check_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIError(f"Backend not reachable at {API_BASE_URL}") from exc


def ask(question: str, history: list[dict] | None = None) -> dict:
    """POST /query -> {answer, sources, chunks, model, elapsed_ms}."""
    payload = {"question": question}
    if history:
        payload["history"] = history

    try:
        response = requests.post(
            f"{API_BASE_URL}/query", json=payload, timeout=TIMEOUT
        )
    except requests.Timeout as exc:
        raise APIError("The request timed out. The model may still be loading.") from exc
    except requests.RequestException as exc:
        raise APIError(
            f"Could not reach the backend at {API_BASE_URL}. Is uvicorn running?"
        ) from exc

    if response.status_code == 422:
        raise APIError("Please enter a question of at least 3 characters.")
    if response.status_code >= 500:
        detail = _detail(response)
        raise APIError(f"The server had a problem: {detail}")
    if not response.ok:
        raise APIError(_detail(response))

    return response.json()


def _detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def upload_file(file_bytes: bytes, filename: str) -> dict:
    """POST /upload -> Uploads a document to the backend vector store."""
    try:
        files = {"file": (filename, file_bytes)}
        # تم تغيير الـ timeout إلى UPLOAD_TIMEOUT بدلاً من TIMEOUT العادي
        response = requests.post(
            f"{API_BASE_URL}/upload", files=files, timeout=UPLOAD_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise APIError("رفع ومعالجة الملف أخذت وقتاً طويلاً. الملف كبير وجاري معالجته في الخلفية.") from exc
    except requests.RequestException as exc:
        response_obj = locals().get("response")
        detail_msg = _detail(response_obj) if response_obj is not None else str(exc)
        raise APIError(f"Failed to upload file: {detail_msg}") from exc