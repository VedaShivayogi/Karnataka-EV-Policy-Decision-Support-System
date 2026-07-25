"""
llm_client.py
A single wrapper so every module can call `chat(prompt, system)` without
caring whether the backend is Groq, Gemini, or a local Ollama model.
This is what makes the "free API" swap-in painless.
"""

import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def chat(prompt: str, system: str = "You are a helpful assistant.", temperature: float = 0.3) -> str:
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        return _chat_groq(prompt, system, temperature)
    elif provider == "gemini":
        return _chat_gemini(prompt, system, temperature)
    elif provider == "ollama":
        return _chat_ollama(prompt, system, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _chat_groq(prompt, system, temperature):
    if not config.GROQ_API_KEY:
        return "[GROQ_API_KEY not set — get a free key at https://console.groq.com]"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": config.GROQ_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _chat_gemini(prompt, system, temperature):
    if not config.GOOGLE_API_KEY:
        return "[GOOGLE_API_KEY not set — get a free key at https://aistudio.google.com]"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}:generateContent?key={config.GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _chat_ollama(prompt, system, temperature):
    url = f"{config.OLLAMA_HOST}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]
