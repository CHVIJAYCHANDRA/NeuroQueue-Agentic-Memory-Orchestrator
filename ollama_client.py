import subprocess
import shlex
import os
import sys
from typing import Optional

LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")

def call_ollama_prompt(prompt: str, model: Optional[str]=None, timeout: int = 300) -> str:
    model = model or LLM_MODEL
    cmd = f"ollama run {shlex.quote(model)}"
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.run(
            shlex.split(cmd),
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            timeout=timeout
        )
        if proc.returncode != 0:
            return f"[OLLAMA ERROR] returncode={proc.returncode}\n{proc.stderr}\n{proc.stdout}"
        output = proc.stdout.strip()
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        output = output.replace('\ufffd', '')
        return output
    except subprocess.TimeoutExpired:
        return "[OLLAMA ERROR] Timeout"
    except FileNotFoundError:
        return "[OLLAMA ERROR] 'ollama' not found — please install Ollama and pull the model (see README)."

