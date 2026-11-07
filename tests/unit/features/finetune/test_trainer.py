"""Unit tests for PEFT trainer wiring (no GPU / training stack required)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

from features.finetune.models import TrainingOptions
from features.finetune.trainer import _compute_warmup_steps, _train_peft


def _options(**overrides: Any) -> TrainingOptions:
    values: dict[str, Any] = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "method": "qlora",
        "backend": "peft",
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "warmup_ratio": 0.03,
    }
    values.update(overrides)
    return TrainingOptions(**values)


def test_compute_warmup_steps_matches_requested_formula() -> None:
    """20 examples, batch 2, accum 8, 3 epochs, ratio 0.03 → 1 step."""
    # total_steps = (20 // (2 * 8)) * 3 = 3; warmup = max(1, int(3 * 0.03)) = 1
    assert _compute_warmup_steps(dataset_size=20, options=_options()) == 1


def test_compute_warmup_steps_full_ratio() -> None:
    """warmup_ratio=1.0 covers every training step."""
    options = _options(warmup_ratio=1.0)
    assert _compute_warmup_steps(dataset_size=20, options=options) == 3


def test_compute_warmup_steps_never_zero() -> None:
    """Even a tiny dataset/ratio still yields at least one warmup step."""
    options = _options(epochs=1, warmup_ratio=0.0)
    assert _compute_warmup_steps(dataset_size=4, options=options) == 1


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, _name: str) -> _FakeTokenizer:
        return cls()

    def save_pretrained(self, _path: str) -> None:
        return None


class _FakeModel:
    def __init__(self) -> None:
        self.to_calls: list[object] = []

    def to(self, target: object) -> _FakeModel:
        self.to_calls.append(target)
        return self


def _install_fake_stack(captured: dict[str, Any]) -> dict[str, ModuleType]:
    """Build in-memory torch/peft/transformers/trl stand-ins."""
    torch_mod = ModuleType("torch")
    torch_mod.bfloat16 = "bf16"  # type: ignore[attr-defined]

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["lora"] = kwargs

    class FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    def fake_get_peft_model(model: _FakeModel, _config: object) -> _FakeModel:
        captured["peft_wrapped"] = True
        return model

    peft_mod = ModuleType("peft")
    peft_mod.LoraConfig = FakeLoraConfig  # type: ignore[attr-defined]
    peft_mod.TaskType = FakeTaskType  # type: ignore[attr-defined]
    peft_mod.get_peft_model = fake_get_peft_model  # type: ignore[attr-defined]

    class FakeBnB:
        def __init__(self, **kwargs: Any) -> None:
            captured["bnb"] = kwargs
            self.kwargs = kwargs

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> _FakeModel:
            captured["from_pretrained"] = {"name": name, **kwargs}
            model = _FakeModel()
            captured["model"] = model
            return model

    transformers_mod = ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    transformers_mod.BitsAndBytesConfig = FakeBnB  # type: ignore[attr-defined]

    class FakeSFTConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["sft_config"] = kwargs

    class FakeSFTTrainer:
        def __init__(self, **kwargs: Any) -> None:
            captured["trainer"] = kwargs

        def train(self) -> SimpleNamespace:
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def save_model(self, path: str) -> None:
            captured["saved"] = path

    trl_mod = ModuleType("trl")
    trl_mod.SFTConfig = FakeSFTConfig  # type: ignore[attr-defined]
    trl_mod.SFTTrainer = FakeSFTTrainer  # type: ignore[attr-defined]

    return {
        "torch": torch_mod,
        "peft": peft_mod,
        "transformers": transformers_mod,
        "trl": trl_mod,
    }


def _run_peft(
    tmp_path: Any,
    *,
    method: str,
    captured: dict[str, Any],
) -> None:
    dataset: list[dict[str, Any]] = [{"messages": []} for _ in range(20)]
    modules = _install_fake_stack(captured)
    with patch.dict(sys.modules, modules):
        _train_peft(
            options=_options(method=method),
            dataset=dataset,
            output_dir=tmp_path,
        )


def test_train_peft_qlora_pins_cuda_and_vram_flags(tmp_path: Any) -> None:
    """QLoRA loads on cuda:0, skips .to(), and uses the VRAM-safe SFTConfig."""
    captured: dict[str, Any] = {}
    _run_peft(tmp_path, method="qlora", captured=captured)

    loaded = captured["from_pretrained"]
    assert loaded["device_map"] == "cuda:0"
    assert loaded["quantization_config"] is not None
    assert "backend" not in loaded
    assert captured["model"].to_calls == []
    assert "llm_int8_enable_fp32_cpu_offload" not in captured["bnb"]

    sft = captured["sft_config"]
    assert "warmup_ratio" not in sft
    assert sft["warmup_steps"] == 1
    assert sft["gradient_checkpointing"] is True
    assert sft["optim"] == "paged_adamw_8bit"
    assert sft["fp16"] is False
    assert sft["bf16"] is True
    assert sft["report_to"] == "none"
    assert sft["num_train_epochs"] == 3
    assert sft["per_device_train_batch_size"] == 2
    assert sft["gradient_accumulation_steps"] == 8
    assert sft["max_length"] == 4096

    trainer_kwargs = captured["trainer"]
    assert "tokenizer" not in trainer_kwargs
    assert "processing_class" in trainer_kwargs


def test_train_peft_lora_moves_model_to_cuda_bf16(tmp_path: Any) -> None:
    """Non-quantized LoRA still pins cuda:0 and then casts the model to bf16."""
    captured: dict[str, Any] = {}
    _run_peft(tmp_path, method="lora", captured=captured)

    loaded = captured["from_pretrained"]
    assert loaded["device_map"] == "cuda:0"
    assert loaded["quantization_config"] is None
    assert captured["model"].to_calls == ["cuda:0", "bf16"]
    assert "bnb" not in captured
