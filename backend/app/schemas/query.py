"""Request / response models for the /query endpoint."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's question, answered only from the indexed documents.",
    )
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional conversation history list containing prior 'role' and 'content' dicts.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What wash temperature is recommended for cotton?",
                    "history": [
                        {"role": "user", "content": "Hi, I have a Samsung washing machine."},
                        {"role": "assistant", "content": "Hello! How can I help with your Samsung washing machine?"}
                    ]
                }
            ]
        }
    }


class SourceChunk(BaseModel):
    """One retrieved chunk, returned so the UI can show the grounding."""
    label: str                 # e.g. "samsung_ww90.pdf p.12"
    document: str
    page: int | None = None
    score: float               # cosine distance: lower = closer
    preview: str               # first ~300 chars of the chunk


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]                       # required by the spec
    chunks: list[SourceChunk] = []           # extra detail for the frontend
    model: str
    elapsed_ms: int