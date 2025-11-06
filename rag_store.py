import os
import pickle
import numpy as np
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class RAGStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_path: Optional[str] = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for RAG. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.persist_path = persist_path or ".rag_store"
        self.encoder = SentenceTransformer(model_name)
        self.documents: List[Dict] = []
        self.index = None
        self.embedding_dim = 384
        
        if os.path.exists(self.persist_path):
            self.load()
        else:
            self._init_index()
    
    def _init_index(self):
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = None
    
    def add_documents(self, documents: List[Dict[str, str]]):
        for doc in documents:
            doc_id = len(self.documents)
            doc["id"] = doc_id
            self.documents.append(doc)
            
            text = doc.get("content", doc.get("text", ""))
            embedding = self.encoder.encode([text])[0]
            
            if FAISS_AVAILABLE and self.index is not None:
                embedding = embedding.astype('float32').reshape(1, -1)
                self.index.add(embedding)
            elif self.index is None:
                self._init_index()
                if FAISS_AVAILABLE and self.index is not None:
                    embedding = embedding.astype('float32').reshape(1, -1)
                    self.index.add(embedding)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.documents:
            return []
        
        query_embedding = self.encoder.encode([query])[0]
        
        if FAISS_AVAILABLE and self.index is not None:
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    doc = self.documents[int(idx)].copy()
                    doc["similarity_score"] = float(1 / (1 + dist))
                    results.append(doc)
            return results
        else:
            query_emb = query_embedding
            doc_embeddings = np.array([self.encoder.encode([doc.get("content", doc.get("text", ""))])[0] 
                                      for doc in self.documents])
            
            similarities = np.dot(doc_embeddings, query_emb) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                doc = self.documents[int(idx)].copy()
                doc["similarity_score"] = float(similarities[idx])
                results.append(doc)
            return results
    
    def save(self):
        os.makedirs(self.persist_path, exist_ok=True)
        with open(os.path.join(self.persist_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, os.path.join(self.persist_path, "index.faiss"))
    
    def load(self):
        doc_path = os.path.join(self.persist_path, "documents.pkl")
        if os.path.exists(doc_path):
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
        
        index_path = os.path.join(self.persist_path, "index.faiss")
        if FAISS_AVAILABLE and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self._init_index()
            if self.documents:
                embeddings = np.array([self.encoder.encode([doc.get("content", doc.get("text", ""))])[0] 
                                      for doc in self.documents])
                if FAISS_AVAILABLE and self.index is not None:
                    embeddings = embeddings.astype('float32')
                    self.index.add(embeddings)
    
    def clear(self):
        self.documents = []
        if os.path.exists(self.persist_path):
            import shutil
            shutil.rmtree(self.persist_path)
        self._init_index()

