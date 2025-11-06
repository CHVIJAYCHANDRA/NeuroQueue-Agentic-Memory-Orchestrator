import os
import sys
from dotenv import load_dotenv
from fifo_memory import FIFOMemory
from agents import ListenerAgent, PlannerAgent, AnalystAgent, WriterAgent, CrewController
from rag_store import RAGStore

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

load_dotenv()

def build_controller(parallel: bool = False, enable_rag: bool = True):
    fifo_size = int(os.getenv("FIFO_SIZE", 5))
    enable_salience = os.getenv("ENABLE_SALIENCE", "true").lower() == "true"
    memory = FIFOMemory(max_size=fifo_size, enable_salience=enable_salience)
    
    rag_store = None
    if enable_rag:
        try:
            rag_store = RAGStore()
        except Exception as e:
            print(f"Warning: RAG store initialization failed: {e}. Continuing without RAG.")
    
    agents = [
        ListenerAgent(memory),
        PlannerAgent(memory),
        AnalystAgent(memory, rag_store=rag_store),
        WriterAgent(memory)
    ]
    controller = CrewController(memory, agents, parallel=parallel)
    return controller

def interactive_loop():
    parallel_mode = os.getenv("PARALLEL_MODE", "false").lower() == "true"
    controller = build_controller(parallel=parallel_mode)
    mode_str = "PARALLEL" if parallel_mode else "SEQUENTIAL"
    print(f"NeuroQueue Agentic Memory Orchestrator ({mode_str} mode) — type 'exit' to quit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit","quit"):
            print("Goodbye")
            break
        docs = ""
        result = controller.run_workflow(user_input, docs=docs)
        print("\n--- Agent Outputs ---")
        for name, out in result["agent_outputs"].items():
            print(f"\n[{name}]\n{out}\n")
        if result.get("consensus"):
            print(f"\n[Consensus]\n{result['consensus']}\n")
        print("--- Memory snapshot (JSON) ---")
        print(result["memory"])
        print("\n-------------------------\n")

if __name__ == "__main__":
    interactive_loop()

