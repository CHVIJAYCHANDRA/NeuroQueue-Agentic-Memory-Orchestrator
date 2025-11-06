import time
import json
from typing import List, Dict, Optional

class FIFOMemory:
    def __init__(self, max_size:int = 5, enable_salience: bool = True):
        self.max_size = max_size
        self.enable_salience = enable_salience
        self.queue = []

    def _calculate_salience(self, text: str) -> float:
        if not self.enable_salience:
            return 0.5
        keywords = ["important", "critical", "key", "essential", "significant", "crucial", "vital"]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        length_score = min(len(text) / 500, 1.0)
        salience = 0.3 + (keyword_count * 0.1) + (length_score * 0.2)
        return min(salience, 1.0)

    def add_item(self, role: str, text: str, salience: Optional[float] = None):
        if salience is None:
            salience = self._calculate_salience(text)
        
        entry = {
            "ts": time.time(),
            "role": role,
            "text": text,
            "salience": salience
        }
        self.queue.append(entry)
        
        if len(self.queue) > self.max_size:
            self._evict_lowest_salience()

    def _evict_lowest_salience(self):
        if not self.enable_salience or len(self.queue) <= self.max_size:
            return
        
        self.queue.sort(key=lambda x: x.get("salience", 0.5))
        while len(self.queue) > self.max_size:
            self.queue.pop(0)
        
        self.queue.sort(key=lambda x: x.get("ts", 0))

    def get_context_text(self) -> str:
        sorted_queue = sorted(self.queue, key=lambda x: x.get("ts", 0))
        return "\n\n".join([f"{e['role']}: {e['text']}" for e in sorted_queue])

    def get_items(self) -> List[Dict]:
        return sorted(self.queue, key=lambda x: x.get("ts", 0))

    def clear(self):
        self.queue.clear()

    def set_size(self, n: int):
        self.max_size = n
        if len(self.queue) > n:
            self.queue.sort(key=lambda x: x.get("salience", 0.5) if self.enable_salience else x.get("ts", 0), reverse=True)
            self.queue = self.queue[:n]
            self.queue.sort(key=lambda x: x.get("ts", 0))

    def export_json(self) -> str:
        return json.dumps(self.get_items(), indent=2)

