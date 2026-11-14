"""Фиксированный протокол ChatML для HF, GGUF и Ollama."""

import hashlib
from pathlib import Path

# Точные варианты не позволяют незаметно поменять один из двух форматов.
ROOT = Path(__file__).resolve().parents[2]
JINJA = ROOT / "infra/llm/prompts/finetune_chat.jinja"
OLLAMA = ROOT / "infra/llm/prompts/finetune_ollama.gotmpl"


def check_template_pair(jinja: Path, ollama: Path) -> None:
    expected = (
        "8cbee9db84ee2ff363e26e8eca4ce98920c2b3b008e47412038e274c6516b65e",
        "235e8905cc63bc53b15a73b1e593623c2f246dad8708ed44042864f30b86e6ae",
    )
    actual = tuple(
        hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for path in (jinja, ollama)
    )
    if actual != expected:
        raise ValueError("Шаблоны не совпадают с зафиксированной парой ChatML")
