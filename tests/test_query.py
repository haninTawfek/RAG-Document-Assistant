"""Backend tests. They use fake services so pytest runs without Ollama or the store."""
from fastapi.testclient import TestClient

from app.api.routes.query import get_generator, get_retriever
from app.main import app
from app.services.retrieval import Chunk


class FakeRetriever:
    class _Collection:
        def count(self):
            return 3

    collection = _Collection()

    def retrieve(self, question, k=4):
        return [
            Chunk(
                text="Cotton items should be washed at 40 degrees Celsius.",
                document="samsung_ww90.pdf",
                page=12,
                score=0.21,
                chunk_id="c1",
            )
        ]


class FakeGenerator:
    def health(self):
        return True

    def answer(self, question, chunks):
        return "Wash cotton at 40 degrees Celsius. [1]"


app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
app.dependency_overrides[get_generator] = lambda: FakeGenerator()

client = TestClient(app)


def test_health_returns_ok_shape():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_query_happy_path_returns_answer_and_sources():
    """Happy path: a valid question returns a grounded answer plus its sources."""
    response = client.post("/query", json={"question": "What temperature for cotton?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"] == "Wash cotton at 40 degrees Celsius. [1]"
    assert body["sources"] == ["samsung_ww90.pdf p.12"]
    assert body["chunks"][0]["page"] == 12


def test_query_rejects_invalid_input_with_422():
    """Invalid input: an empty question fails validation before reaching the LLM."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_missing_field_with_422():
    response = client.post("/query", json={})
    assert response.status_code == 422
