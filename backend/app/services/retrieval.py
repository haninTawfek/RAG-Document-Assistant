"""Loads the persisted Chroma vector store and retrieves relevant chunks.

The store is built once by notebooks/rag_pipeline.ipynb and copied into
backend/data/vector_store2 — nothing is re-embedded at request time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    document: str
    page: int | None
    score: float
    chunk_id: str

    @property
    def label(self) -> str:
        return f"{self.document} p.{self.page}" if self.page is not None else self.document


class Retriever:
    def __init__(
        self, 
        store_dir: str = "backend/data/vector_store2", 
        collection_name: str = "manuals_v2", 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        store_path = Path(store_dir)
        if not store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_path}. "
                "Run notebooks/rag_pipeline.ipynb and copy the output here."
            )

        # If the notebook exported a config.json, trust it over the defaults so the
        # backend can never embed queries with a different model than the index.
        config_file = store_path / "config.json"
        if config_file.exists():
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            collection_name = cfg.get("collection_name", collection_name)
            embedding_model = cfg.get("embedding_model", embedding_model)
            logger.info("Loaded store config: %s", cfg)

        self.embedding_model = embedding_model
        self.collection_name = collection_name

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        client = chromadb.PersistentClient(path=str(store_path))
        self.collection = client.get_collection(name=collection_name, embedding_function=ef)
        logger.info(
            "Vector store ready: collection=%r chunks=%d model=%r",
            collection_name, self.collection.count(), embedding_model,
        )

    def retrieve(self, question: str, k: int = 8) -> list[Chunk]:
        res = self.collection.query(query_texts=[question], n_results=k)
        chunks: list[Chunk] = []
    
        for text, meta, dist, cid in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]
        ):
            # استبعاد القطع التي تحتوي على ضوضاء أو distance مرتفع جداً
            if dist > 0.60:
                continue
            
            page = meta.get("page")
            chunks.append(
                Chunk(
                    text=text,
                    document=str(meta.get("source", "unknown")),
                    page=int(page) if page is not None else None,
                    score=float(dist),
                    chunk_id=str(cid),
                )
            )
        return chunks

    def add_pdf_document(self, file_path: str, chunk_size: int = 1500, chunk_overlap: int = 150):
        """تقطيع ملف PDF جديد وإضافته فوراً للـ Collection المفتوحة في ChromaDB"""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        logger.info(f"Indexing new file: {file_path}")
        
        # 1. تحميل الملف
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 2. تقطيع النص
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # 3. إعداد البيانات للـ Vector Store
        documents = [doc.page_content for doc in splits]
        metadatas = [
            {
                "source": Path(file_path).name,
                "page": doc.metadata.get("page", 0) + 1
            }
            for doc in splits
        ]
        # إنشاء IDs فريدة لكل Chunk
        ids = [f"{Path(file_path).stem}_chunk_{i}" for i in range(len(splits))]

        # 4. الحفظ في ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added {len(documents)} new chunks to collection {self.collection_name}.")