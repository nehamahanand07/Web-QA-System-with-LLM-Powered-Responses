import os, json, faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleVectorStore:
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Dict] = []

    def _index_path(self): return os.path.join(self.index_dir, "index.faiss")
    def _meta_path(self): return os.path.join(self.index_dir, "meta.json")

    def build(self, docs: List[Dict]):
        texts = [d["content"] for d in docs]
        X = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
        self.meta = docs

    def save(self):
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def load(self):
        self.index = faiss.read_index(self._index_path())
        with open(self._meta_path(), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def search(self, query: str, top_k=5):
        qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1: continue
            d = self.meta[idx]
            results.append({**d, "score": float(score)})
        return results
