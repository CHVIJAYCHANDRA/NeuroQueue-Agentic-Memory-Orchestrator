from typing import List, Dict, Any, Optional
from ollama_client import call_ollama_prompt
from fifo_memory import FIFOMemory

class BaseAgent:
    def __init__(self, name: str, memory: FIFOMemory):
        self.name = name
        self.memory = memory

    def prompt_wrapper(self, user_input: str, extra_context: str = "") -> str:
        base = f"[Agent: {self.name}]\n"
        memory_text = self.memory.get_context_text()
        if memory_text:
            base += f"\n--- FIFO Memory (oldest->newest) ---\n{memory_text}\n"
        if extra_context:
            base += f"\n--- Extra Context ---\n{extra_context}\n"
        base += f"\n--- User Input ---\n{user_input}\n\nProvide a concise, useful response (3-6 sentences)."
        return base

    def run(self, user_input: str, **kwargs) -> str:
        prompt = self.prompt_wrapper(user_input, kwargs.get("extra_context",""))
        resp = call_ollama_prompt(prompt)
        if resp:
            self.memory.add_item(self.name, resp)
        return resp

class ListenerAgent(BaseAgent):
    def __init__(self, memory: FIFOMemory):
        super().__init__("ListenerAgent", memory)

    def run(self, user_input: str, **kwargs) -> str:
        self.memory.add_item("User", user_input)
        return f"Captured: {user_input[:200]}"

class PlannerAgent(BaseAgent):
    def __init__(self, memory: FIFOMemory):
        super().__init__("PlannerAgent", memory)

    def run(self, user_input: str, **kwargs) -> str:
        prompt = self.prompt_wrapper(user_input) + "\n\nTask: Break the user's request into 3 ordered sub-tasks (bullet list)."
        resp = call_ollama_prompt(prompt)
        self.memory.add_item(self.name, resp)
        return resp

class AnalystAgent(BaseAgent):
    def __init__(self, memory: FIFOMemory, rag_store: Optional[Any] = None):
        super().__init__("AnalystAgent", memory)
        self.rag_store = rag_store

    def run(self, user_input: str, **kwargs) -> str:
        docs = kwargs.get("docs", "")
        
        if self.rag_store:
            retrieved_docs = self.rag_store.search(user_input, top_k=3)
            if retrieved_docs:
                rag_context = "\n--- Retrieved Documents (RAG) ---\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    content = doc.get("content", doc.get("text", ""))
                    score = doc.get("similarity_score", 0.0)
                    rag_context += f"\n[Doc {i}, Score: {score:.3f}]\n{content[:300]}\n"
                docs = (docs + "\n" + rag_context) if docs else rag_context
        
        prompt = self.prompt_wrapper(user_input, extra_context=(docs or ""))
        prompt += "\n\nTask: Extract 2–3 key insights or facts relevant to the user request."
        resp = call_ollama_prompt(prompt)
        self.memory.add_item(self.name, resp)
        
        if self.rag_store and resp:
            self.rag_store.add_documents([{"content": resp, "role": self.name, "source": "agent_output"}])
            self.rag_store.save()
        
        return resp

class WriterAgent(BaseAgent):
    def __init__(self, memory: FIFOMemory):
        super().__init__("WriterAgent", memory)

    def run(self, user_input: str, **kwargs) -> str:
        prompt = self.prompt_wrapper(user_input, kwargs.get("extra_context",""))
        prompt += "\n\nTask: Write a concise final summary in 4–8 sentences that synthesizes the above."
        resp = call_ollama_prompt(prompt)
        self.memory.add_item(self.name, resp)
        return resp

class CrewController:
    def __init__(self, memory: FIFOMemory, agents: List[BaseAgent], parallel: bool = False):
        self.memory = memory
        self.agents = agents
        self.parallel = parallel

    def _run_agent(self, agent: BaseAgent, user_input: str, docs: str = "") -> tuple:
        try:
            if isinstance(agent, AnalystAgent):
                out = agent.run(user_input, docs=docs)
            else:
                out = agent.run(user_input)
            return agent.name, out
        except Exception as e:
            return agent.name, f"[ERROR] {str(e)}"

    def _consensus_vote(self, outputs: Dict[str, str]) -> str:
        if len(outputs) < 2:
            return list(outputs.values())[0] if outputs else ""
        
        responses = list(outputs.values())
        if len(set(responses)) == 1:
            return responses[0]
        
        prompt = "Given these agent responses, provide a consensus synthesis:\n\n"
        for name, response in outputs.items():
            prompt += f"[{name}]: {response}\n\n"
        prompt += "Provide a unified, balanced consensus response."
        
        consensus = call_ollama_prompt(prompt)
        return consensus or responses[0]

    def run_workflow(self, user_input: str, docs: str = "") -> Dict[str, Any]:
        outputs = {}
        
        if self.parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_agent = {
                    executor.submit(self._run_agent, agent, user_input, docs): agent
                    for agent in self.agents
                }
                
                for future in as_completed(future_to_agent):
                    agent_name, result = future.result()
                    outputs[agent_name] = result
        else:
            for agent in self.agents:
                if isinstance(agent, AnalystAgent):
                    out = agent.run(user_input, docs=docs)
                else:
                    out = agent.run(user_input)
                outputs[agent.name] = out
        
        consensus = None
        if self.parallel and len(outputs) > 1:
            consensus = self._consensus_vote(outputs)
            if consensus:
                self.memory.add_item("Consensus", consensus, salience=0.9)
        
        return {
            "user_input": user_input,
            "agent_outputs": outputs,
            "consensus": consensus,
            "memory": self.memory.export_json()
        }

