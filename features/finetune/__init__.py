"""Offline fine-tune pipeline: dataset preparation and LoRA/QLoRA training.

The package is split into pure building blocks and a thin CLI layer:

- :mod:`features.finetune.models` — immutable dataset/training models;
- :mod:`features.finetune.game_reconstructor` — rebuild :class:`GameState`
  from persisted events at a prediction timestamp;
- :mod:`features.finetune.dataset_builder` — one SFT example per prediction;
- :mod:`features.finetune.anonymizer` — ``player_N`` pseudonymisation;
- :mod:`features.finetune.trainer` — LoRA/QLoRA runner (Unsloth / PEFT).

Nothing here runs during live gameplay: the pipeline is invoked from
``scripts/prepare_finetune_dataset.py`` and ``scripts/finetune_lora.py``.
"""
