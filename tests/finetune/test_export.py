"""Проверка экспорта без GPU и настоящих весов."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.finetune.common import digest
from scripts.finetune.export import export
from scripts.finetune.template import JINJA, OLLAMA


def test_export_mock_conversion(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    template = adapter / "chat_template.jinja"
    template.write_bytes(JINJA.read_bytes())
    (adapter / "quality.json").write_text(
        json.dumps(
            {
                "ready_for_export": True,
                "ready_for_use": False,
                "status": "experimental",
                "metrics": {
                    "validation_completed": True,
                    "training_steps": 1,
                    "validation_history": [1.9, 1.5],
                    "train_loss": 1.2,
                    "test_loss": 1.6,
                },
                "risks": [],
                "base_model": "test-base",
                "template_sha256": digest(template),
            }
        ),
        encoding="utf-8",
    )
    cpp = tmp_path / "cpp"
    cpp.mkdir()
    (cpp / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    config = {
        "adapter_dir": str(adapter),
        "base_model": "test-base",
        "chat_template": str(JINJA),
        "ollama_template": str(OLLAMA),
        "export_dir": str(tmp_path / "out"),
        "llama_cpp_dir": str(cpp),
        "max_val_loss": 2.0,
        "max_val_train_gap": 0.5,
        "max_val_loss_increase": 0.2,
    }

    class FakeModel:
        def merge_and_unload(self):
            return self

        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTokenizer:
        chat_template = ""

        def save_pretrained(self, path):
            assert self.chat_template == template.read_text(encoding="utf-8")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeModel()
            ),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel())
        ),
    )

    def run(command, check):
        assert check and command[-2:] == ["--outtype", "f16"]
        Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF mock")

    monkeypatch.setattr("scripts.finetune.export.subprocess.run", run)
    export(config)
    modelfile = (tmp_path / "out/Modelfile").read_text(encoding="utf-8")
    assert "FROM ./mafia-f16.gguf" in modelfile
    assert OLLAMA.read_text(encoding="utf-8").rstrip("\n") in modelfile
    assert (tmp_path / "out/merged/chat_template.jinja").read_bytes() == JINJA.read_bytes()

    (adapter / "quality.json").write_text(
        json.dumps({"ready_for_export": False, "risks": ["loss"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="loss"):
        export(config)
    assert not (tmp_path / "out/Modelfile").exists()


def test_export_rejects_unvalidated_artifact(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "quality.json").write_text(
        json.dumps(
            {
                "ready_for_export": True,
                "ready_for_use": False,
                "status": "experimental",
                "risks": [],
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss"):
        export({"adapter_dir": str(adapter), "export_dir": str(tmp_path / "out")})
