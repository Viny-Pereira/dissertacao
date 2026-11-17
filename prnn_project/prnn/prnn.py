"""
prnn.py

Concrete implementations of the Physically Recurrent Neural Network (PRNN) architecture.
Maintains strict separation of concerns via dependency injection:
  - `PRNNCell`: encapsulates single-step localization, multi-phase constitutive physics,
    and homogenization.
  - `PRNNSequence`: wraps any `PRNNCell` to provide temporal sequence unrolling (`forward`),
    latent field extraction (`get_local_states`), and online Jem/Jive solver callbacks
    (`evaluate`, `commit`, `cancel`, `observe`, `reset`).
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from .interfaces import Decoder, Encoder, PRNN
from .layers import DecoderFactory, EncoderFactory
from .material import MaterialLayer, MaterialLayerContext, MaterialSpec


class PRNNCell(nn.Module):
    """
    Executes a single constitutive step of the Physically Recurrent Neural Network:
        1. Localization (Encoder): macro-strain -> fictitious micro-strains.
        2. Physics (MaterialLayer): integrates multi-phase constitutive laws via a stateful context.
        3. Homogenization (Decoder): micro-stresses -> macro-stress.
    """

    def __init__(
        self,
        num_material_points: int,
        materials: MaterialSpec,
        dim: Optional[int] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        tensor_components: Optional[int] = None,
        encoder_type: str = "linear",
        decoder_type: str = "dense",
        normalize_weights: Optional[bool] = None,
        history_step: float = 1.0e-7,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initializes the single-step PRNN cell.

        Args:
            num_material_points: Total number of fictitious material points per sample.
            materials: Phase specification `[(model_or_factory, count), ...]` or a single
                `ConstitutiveModel` / Jive `MaterialFactory`.
            dim: Spatial dimension (2 -> 3 Voigt components, 3 -> 6 Voigt components).
                Optional if `input_size` and `output_size` are provided explicitly.
            input_size: Macroscopic strain vector dimension.
            output_size: Macroscopic stress vector dimension.
            tensor_components: Local Voigt stress/strain dimension.
            encoder_type: Localization layer key ('linear', 'consistent' / 'tc').
            decoder_type: Homogenization layer key ('dense', 'sparse', 'point',
                'consistent' / 'tc', plus '*_normalized' variants).
            normalize_weights: Optional boolean override for decoder weight normalization.
            history_step: Perturbation step for Jive C++ finite-difference history VJPs.
            device: PyTorch computation device. Defaults to CPU.
            dtype: Floating-point precision. Defaults to `torch.float64` (required by Jive).
        """
        super().__init__()
        if dim is not None:
            if dim == 2:
                default_components = 3
            elif dim == 3:
                default_components = 6
            else:
                raise ValueError(f"Invalid spatial dimension '{dim}'. Expected 2 or 3.")

            self.input_size = input_size if input_size is not None else default_components
            self.output_size = output_size if output_size is not None else default_components
            self.tensor_components = (
                tensor_components if tensor_components is not None else default_components
            )
        else:
            if input_size is None or output_size is None:
                raise ValueError(
                    "Must specify spatial 'dim' (e.g., dim=3) OR both 'input_size' and 'output_size'."
                )
            self.input_size = input_size
            self.output_size = output_size
            self.tensor_components = (
                tensor_components if tensor_components is not None else input_size
            )

        enc_norm = encoder_type.lower()
        dec_norm = decoder_type.lower()
        if dec_norm in ("consistent", "tc") and enc_norm not in ("consistent", "tc"):
            raise ValueError(
                f"Thermodynamic mismatch: decoder '{decoder_type}' requires "
                f"encoder_type='consistent' (got '{encoder_type}')."
            )

        self.dim = dim
        self.num_material_points = num_material_points
        self.latent_size = self.num_material_points * self.tensor_components
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        # 1. Physics: Multi-phase MaterialLayer (Python ConstitutiveModel or Jive C++ factory)
        self.material_layer = MaterialLayer(
            materials=materials,
            components=self.tensor_components,
            n_matpts=self.num_material_points,
            history_step=history_step,
        )

        # 2. Encoder (Localization: macro_strain -> micro_strain)
        self.encoder: Encoder = EncoderFactory.create(
            encoder_type=encoder_type,
            in_features=self.input_size,
            out_features=self.latent_size,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        # 3. Decoder (Homogenization: micro_stress -> macro_stress)
        decoder_kwargs: Dict[str, Any] = {}
        if normalize_weights is not None:
            decoder_kwargs["normalize_weights"] = normalize_weights

        self.decoder: Decoder = DecoderFactory.create(
            decoder_type=decoder_type,
            in_features=self.latent_size,
            out_features=self.output_size,
            bias=False,
            device=self.device,
            dtype=self.dtype,
            **decoder_kwargs,
        )

    def forward(
        self,
        macro_strain_step: Tensor,
        context: MaterialLayerContext,
    ) -> Tensor:
        """
        Computes the macroscopic trial stress for a single time increment using an
        active `MaterialLayerContext`.

        Args:
            macro_strain_step: Macroscopic strain tensor of shape [batch_size, input_size].
            context: Active stateful `MaterialLayerContext` managing history states.

        Returns:
            Macroscopic trial stress tensor of shape [batch_size, output_size].
        """
        batch_size = macro_strain_step.size(0)
        total_points = batch_size * self.num_material_points

        # 1. Localization
        micro_strain_flat = self.encoder(macro_strain_step)

        # 2. Physics / Constitutive Evaluation
        micro_strain_reshaped = micro_strain_flat.reshape(total_points, self.tensor_components)
        micro_stress_reshaped = context.update(micro_strain_reshaped).to(dtype=self.dtype)

        # 3. Homogenization (passes E_raw explicitly for Hill-Mandel consistent decoders)
        micro_stress_flat = micro_stress_reshaped.reshape(batch_size, self.latent_size)
        e_raw = getattr(self.encoder, "E_raw", None)
        return self.decoder(micro_stress_flat, e_raw=e_raw)


class PRNNSequence(PRNN):
    """
    Wraps an injected `PRNNCell` to handle:
      - Temporal sequence unrolling for offline training (`forward`),
      - Latent field extraction for validation and plotting (`get_local_states`),
      - Step-by-step online trial/commit/cancel execution for Jem/Jive (`evaluate`,
        `commit`, `cancel`, `observe`, `reset`),
      - Self-describing checkpoint serialization (`get_config`, `save_checkpoint`, `from_config`).
    """

    def __init__(self, cell: PRNNCell):
        """
        Initializes the sequence and online driver wrapper around a configured `PRNNCell`.

        Args:
            cell: Instantiated `PRNNCell` defining the single-step network topology and physics.
        """
        super().__init__()
        self.cell = cell

        # Online Jive driver state
        self._online: Optional[MaterialLayerContext] = None
        self._online_count: Optional[int] = None
        self._trial_input: Optional[Tensor] = None

    def forward(self, macro_strain_sequence: Tensor) -> Tensor:
        """
        Offline training forward pass: initializes a fresh independent material history
        per batched loading path and unrolls across the temporal sequence.

        Args:
            macro_strain_sequence: Macroscopic strain sequence [batch_size, seq_len, input_size].

        Returns:
            Predicted macroscopic stress sequence [batch_size, seq_len, output_size].
        """
        if (
            macro_strain_sequence.ndim != 3
            or macro_strain_sequence.shape[2] != self.cell.input_size
            or macro_strain_sequence.shape[1] == 0
        ):
            raise ValueError(
                f"Expected input tensor of shape [batch, nonempty_seq_len, {self.cell.input_size}], "
                f"got {tuple(macro_strain_sequence.shape)}."
            )

        batch_size, seq_len, _ = macro_strain_sequence.shape
        context = self.cell.material_layer.create_context(batch_size=batch_size, training=True)

        stress_steps = []
        for t in range(seq_len):
            current_strain = macro_strain_sequence[:, t, :]
            current_stress = self.cell(current_strain, context)
            context.commit()
            stress_steps.append(current_stress)

        return torch.stack(stress_steps, dim=1)

    @torch.no_grad()
    def get_local_states(
        self, macro_strain_sequence: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Unrolls a strain sequence in evaluation mode while recording macroscopic stress,
        fictitious micro-strains, micro-stresses, and internal material history variables
        for validation and micromechanical post-processing.

        Args:
            macro_strain_sequence: Strain sequence of shape [batch_size, seq_len, input_size].

        Returns:
            Tuple `(macro_stress_seq, local_strains_seq, local_stresses_seq, local_history_seq)`:
              - `macro_stress_seq`: [batch_size, seq_len, output_size]
              - `local_strains_seq`: [batch_size, seq_len, latent_size]
              - `local_stresses_seq`: [batch_size, seq_len, latent_size]
              - `local_history_seq`: [batch_size, seq_len, num_material_points]
        """
        batch_size, seq_len, num_components = macro_strain_sequence.shape
        device = macro_strain_sequence.device
        dtype = macro_strain_sequence.dtype

        num_points = self.cell.num_material_points
        latent_size = self.cell.latent_size

        macro_stress_seq = torch.zeros(
            (batch_size, seq_len, self.cell.output_size), device=device, dtype=dtype
        )
        local_strains_seq = torch.zeros((batch_size, seq_len, latent_size), device=device, dtype=dtype)
        local_stresses_seq = torch.zeros((batch_size, seq_len, latent_size), device=device, dtype=dtype)
        local_history_seq = torch.zeros((batch_size, seq_len, num_points), device=device, dtype=dtype)

        context = self.cell.material_layer.create_context(batch_size=batch_size, training=False)
        e_raw = getattr(self.cell.encoder, "E_raw", None)

        for t in range(seq_len):
            current_strain = macro_strain_sequence[:, t, :]

            # 1. Localization
            micro_strains = self.cell.encoder(current_strain)
            local_strains_seq[:, t, :] = micro_strains

            # 2. Constitutive Evaluation
            micro_strains_reshaped = micro_strains.reshape(batch_size * num_points, num_components)
            micro_stresses_reshaped = context.update(micro_strains_reshaped)
            context.commit()

            # Extract history per phase if available
            hist_blocks = []
            for ctx, count in context.phase_contexts:
                if hasattr(ctx, "model"):
                    h = ctx.model.get_history().reshape(batch_size, count)
                else:
                    h = torch.zeros((batch_size, count), device=device, dtype=dtype)
                hist_blocks.append(h)
            local_history_seq[:, t, :] = torch.cat(hist_blocks, dim=1)

            micro_stresses = micro_stresses_reshaped.reshape(batch_size, latent_size)
            local_stresses_seq[:, t, :] = micro_stresses

            # 3. Homogenization
            macro_stress_seq[:, t, :] = self.cell.decoder(micro_stresses, e_raw=e_raw)

        return macro_stress_seq, local_strains_seq, local_stresses_seq, local_history_seq

    # ==========================================================================
    # JEM / JIVE ONLINE DRIVER INTERFACE
    # ==========================================================================

    def evaluate(self, strain: Tensor) -> Tensor:
        """
        Online Newton-Raphson trial evaluation: initializes the material context once
        at t=0 and evaluates trial stress without committing history.
        """
        vector = strain.ndim == 1
        batch = strain.unsqueeze(0) if vector else strain
        if batch.ndim != 2 or batch.shape[1] != self.cell.input_size:
            raise ValueError(
                f"Expected strain of shape [{self.cell.input_size}] or [batch, {self.cell.input_size}]."
            )

        batch_size = batch.shape[0]
        count = batch_size * self.cell.num_material_points
        if self._online is None:
            self._online = self.cell.material_layer.create_context(
                batch_size=batch_size, training=False
            )
            self._online_count = count
        if count != self._online_count:
            raise ValueError("Online batch size cannot change; call reset() for a new problem.")

        result = self.cell(batch, self._online)
        self._trial_input = strain.detach().clone()
        return result[0] if vector else result

    def commit(self) -> None:
        """Commits converged internal material states at the end of a load step."""
        if self._online is not None:
            self._online.commit()
        self._trial_input = None

    def cancel(self) -> None:
        """Reverts trial internal material states back to the last committed step."""
        if self._online is not None:
            self._online.cancel()
        self._trial_input = None

    def observe(self, strain: Tensor) -> Tensor:
        """
        Output query for Jive post-processing: evaluates stress at `strain` and
        restores any pending Newton-Raphson trial state afterwards.
        """
        candidate = self._trial_input
        try:
            result = self.evaluate(strain)
            if candidate is None:
                self.cancel()
            else:
                self.evaluate(candidate)
            return result
        except Exception:
            self.cancel()
            raise

    def reset(self) -> None:
        """Resets online material context for a new boundary value problem."""
        self._online = None
        self._online_count = None
        self._trial_input = None

    def script_neural_layers(self) -> "PRNNSequence":
        """
        JIT-compiles the pure neural submodules (`cell.encoder` and `cell.decoder`) for
        faster CPU execution inside Jive callbacks. Falls back gracefully if unsupported.
        """
        try:
            self.cell.encoder = torch.jit.script(self.cell.encoder)
            self.cell.decoder = torch.jit.script(self.cell.decoder)
        except Exception:
            pass
        return self

    # ==========================================================================
    # CHECKPOINT SERIALIZATION HELPERS
    # ==========================================================================

    def get_config(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary capturing the full architectural
        and material phase configuration for self-describing checkpoints.
        """
        return {
            "dim": self.cell.dim,
            "input_size": self.cell.input_size,
            "output_size": self.cell.output_size,
            "tensor_components": self.cell.tensor_components,
            "num_material_points": self.cell.num_material_points,
            "encoder_type": self.cell.encoder_type,
            "decoder_type": self.cell.decoder_type,
            "normalize_weights": getattr(self.cell.decoder, "normalize_weights", None),
            "phases": self.cell.material_layer.describe_phases(),
        }

    def save_checkpoint(self, filename: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves model weights alongside the architectural configuration dictionary.
        """
        payload: Dict[str, Any] = {
            "config": self.get_config(),
            "model_state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, filename)

    @classmethod
    def from_config(cls, config: Dict[str, Any], materials: MaterialSpec) -> "PRNNSequence":
        """
        Reconstructs a `PRNNSequence(PRNNCell(...))` instance from a saved `config`
        dictionary and an active material/factory specification (used by `checkpoint.py`).
        """
        cell = PRNNCell(
            num_material_points=config["num_material_points"],
            materials=materials,
            dim=config.get("dim"),
            input_size=config.get("input_size"),
            output_size=config.get("output_size"),
            tensor_components=config.get("tensor_components"),
            encoder_type=config.get("encoder_type", "linear"),
            decoder_type=config.get("decoder_type", "dense"),
            normalize_weights=config.get("normalize_weights"),
        )
        return cls(cell)