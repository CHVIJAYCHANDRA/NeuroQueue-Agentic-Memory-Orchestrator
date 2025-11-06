import streamlit as st
import sys
import re
import os
from main import build_controller

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

st.set_page_config(page_title="NeuroQueue Agentic Memory Orchestrator", layout="wide")

st.title("NeuroQueue Agentic Memory Orchestrator")

parallel_mode = st.sidebar.checkbox("Parallel Execution", value=False)
enable_rag = st.sidebar.checkbox("Enable RAG", value=True)

controller = build_controller(parallel=parallel_mode, enable_rag=enable_rag)

st.sidebar.header("Controls")

fifo = st.sidebar.slider("FIFO size", 1, 20, int(controller.memory.max_size))

if st.sidebar.button("Apply FIFO size"):
    controller.memory.set_size(fifo)
    st.sidebar.success(f"FIFO size set to {fifo}")

if st.sidebar.button("Clear memory"):
    controller.memory.clear()
    st.sidebar.success("Memory cleared")

st.sidebar.markdown("Make sure `ollama` and the model are installed locally.")

st.write("Enter a prompt and click Run. The agents will run sequentially and outputs are shown below.")

prompt = st.text_input("Your prompt", placeholder="Type your message here...")

def clean_text(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    text = str(text)
    text = re.sub(r'â€¢', '•', text)
    text = re.sub(r'â€"', '—', text)
    text = re.sub(r'â€™', "'", text)
    return text

if st.button("Run", type="primary"):
    mode_text = "in parallel" if parallel_mode else "sequentially"
    with st.spinner(f"Running agents {mode_text}..."):
        result = controller.run_workflow(prompt)
    
    st.subheader("Agent Outputs")
    
    agent_order = ["ListenerAgent", "PlannerAgent", "AnalystAgent", "WriterAgent"]
    
    for agent_name in agent_order:
        if agent_name in result["agent_outputs"]:
            out = clean_text(result["agent_outputs"][agent_name])
            
            with st.expander(f"{agent_name}", expanded=(agent_name == "WriterAgent")):
                st.markdown(out)
    
    if result.get("consensus"):
        st.subheader("Consensus")
        with st.expander("Consensus View", expanded=True):
            st.markdown(clean_text(result["consensus"]))
    
    st.subheader("FIFO Memory State")
    items = controller.memory.get_items()
    
    if items:
        for i, it in enumerate(items):
            text = clean_text(it['text'][:800])
            role = it['role']
            st.markdown(f"**{role}**: {text}")
            if i < len(items) - 1:
                st.divider()
    else:
        st.info("Memory is empty")
    
    st.download_button("Download Memory JSON", data=result["memory"], file_name="memory.json", mime="application/json")

