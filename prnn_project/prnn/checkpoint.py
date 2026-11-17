"""
prnn/checkpoint.py

Checkpoint loader for Jem/Jive embedded C++ drivers and standalone Python inference.
Supports self-describing checkpoints containing a 'config' dictionary (saved via
`PRNNSequence.save_checkpoint` / `Trainer.save`) as well as legacy upstream checkpoints.
"""

from typing import Any, Dict
import torch

from .material import MaterialSpec
from .prnn import PRNNCell, PRNNSequence


def _load_legacy_checkpoint(
    state: Dict[str, torch.Tensor], material_factory: MaterialSpec
) -> PRNNSequence:
    """
    Backward-compatible loader for legacy upstream checkpoints lacking a 'config' entry
    (which stored `fc1.weight` and `fc2.weight` directly at root level).
    """
    features = state["fc1.weight"].shape[1]
    latents = state["fc1.weight"].shape[0]
    outputs = state["fc2.weight"].shape[0]
    if features != outputs or latents % features or latents == 0:
        raise ValueError("Invalid legacy checkpoint feature/material-point dimensions.")

    cell = PRNNCell(
        num_material_points=latents // features,
        materials=material_factory,
        input_size=features,
        output_size=outputs,
        tensor_components=features,
        encoder_type="linear",
        decoder_type="dense",
    )
    model = PRNNSequence(cell).double()

    mapped_state = {
        "cell.encoder.linear.weight": state["fc1.weight"],
        "cell.decoder.weight": state["fc2.weight"],
    }
    if "fc2.bias" in state:
        mapped_state["cell.decoder.bias"] = state["fc2.bias"]

    model.load_state_dict(mapped_state, strict=True)
    return model


def load_model(weights: str, material_factory: MaterialSpec) -> PRNNSequence:
    """
    Entry point invoked by the C++ Jem/Jive driver (or standalone Python scripts)
    to instantiate and restore a trained PRNN from a `.pt` / `.pth` checkpoint.

    Args:
        weights: File path to the saved checkpoint.
        material_factory: Native Jive `MaterialFactory`, pure-Python `ConstitutiveModel`,
            or multi-phase list of `(material_or_factory, count)` tuples.

    Returns:
        Configured `PRNNSequence` in double precision (`float64`), `eval()` mode,
        with gradients disabled and neural layers JIT-scripted for fast inference.
    """
    try:
        checkpoint: Dict[str, Any] = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)

    # Prefer best validation weights when available, falling back to final epoch weights
    if "best_state_dict" in checkpoint and checkpoint["best_state_dict"] is not None:
        state = checkpoint["best_state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if "config" in checkpoint:
        model = PRNNSequence.from_config(
            checkpoint["config"], materials=material_factory
        ).double()
        model.load_state_dict(state, strict=True)
    else:
        model = _load_legacy_checkpoint(state, material_factory)

    model.eval().requires_grad_(False)
    return model.script_neural_layers()