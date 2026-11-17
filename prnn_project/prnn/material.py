"""
Unified physical MaterialLayer and autograd boundary for PRNNs.
Supports multi-phase composite material configurations by default, seamlessly
orchestrating both pure-Python `ConstitutiveModel` instances (from `constitutive.py`)
and native Jem/Jive C++ `MaterialFactory` bridges within a single network.
"""

import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import once_differentiable

from .interfaces import ConstitutiveModel


def _to_numpy(tensor: Tensor):
    """Detaches a CPU float64 tensor and converts it to a contiguous NumPy array."""
    return tensor.detach().contiguous().numpy()


# ==============================================================================
# AUTOGRAD FUNCTIONS FOR NATIVE JEM/JIVE C++ MATERIALS
# ==============================================================================

class MaterialUpdate(torch.autograd.Function):
    """
    Autograd boundary for Jive C++ materials without exposed history variables.
    Replays accepted history for backward differentiation.
    """

    @staticmethod
    def forward(ctx, strain: Tensor, history: Tensor, native: Any, history_step: float) -> Tensor:
        stress, tangent = native.update(_to_numpy(strain))
        ctx.save_for_backward(torch.from_numpy(tangent), strain.detach(), history.detach())
        ctx.native = native
        ctx.history_step = history_step
        return torch.from_numpy(stress)

    @staticmethod
    @once_differentiable
    def backward(ctx, adjoint: Tensor):
        tangent, strain, history = ctx.saved_tensors
        current_grad = torch.einsum("pij,pi->pj", tangent, adjoint)
        history_grad = None
        if ctx.needs_input_grad[1] and history.shape[0]:
            history_grad = torch.from_numpy(
                ctx.native.history_vjp(
                    _to_numpy(history),
                    _to_numpy(strain),
                    _to_numpy(adjoint),
                    ctx.history_step,
                )
            )
        return current_grad, history_grad, None, None


class MaterialStep(torch.autograd.Function):
    """
    Step-wise autograd boundary for Jive C++ materials exposing internal history states.
    Achieves linear O(T) backward complexity in sequence length.
    """

    @staticmethod
    def forward(ctx, strain: Tensor, history: Tensor, native: Any, step: float) -> Tuple[Tensor, Tensor]:
        stress, tangent = native.update(_to_numpy(strain))
        ctx.save_for_backward(torch.from_numpy(tangent), strain.detach(), history.detach())
        ctx.native = native
        ctx.step = step
        return torch.from_numpy(stress), torch.from_numpy(native.history())

    @staticmethod
    @once_differentiable
    def backward(ctx, stress_adjoint: Tensor, history_adjoint: Tensor):
        tangent, strain, history = ctx.saved_tensors
        strain_grad, history_grad = ctx.native.step_vjp(
            _to_numpy(history),
            _to_numpy(strain),
            _to_numpy(stress_adjoint),
            _to_numpy(history_adjoint),
            ctx.step,
        )
        strain_grad = torch.from_numpy(strain_grad)
        strain_grad += torch.einsum("pij,pi->pj", tangent, stress_adjoint)
        return strain_grad, torch.from_numpy(history_grad), None, None


# ==============================================================================
# PHASE CONTEXT ADAPTERS (Jive C++ vs. Pure Python ConstitutiveModel)
# ==============================================================================

class _JivePhaseContext:
    """
    Manages a single phase backed by a native Jem/Jive C++ MaterialFactory.
    """

    def __init__(
        self,
        factory: Callable[[int], Any],
        total_points: int,
        components: int,
        training: bool = False,
        history_step: float = 1.0e-7,
    ):
        self.native = factory(total_points)
        if self.native.components != components:
            raise ValueError(
                f"Network components ({components}) must match configured C++ material ({self.native.components})."
            )
        self.training = training
        self.history_step = history_step
        self.stepwise = training and self.native.history_count >= 0
        self.history: List[Tensor] = []
        self.trial: Optional[Tensor] = None
        if self.stepwise:
            self.state = torch.from_numpy(self.native.history())
            self.trial_state: Optional[Tensor] = None

    def update(self, strain: Tensor) -> Tensor:
        if strain.device.type != "cpu" or strain.dtype != torch.float64:
            raise ValueError("The Jem/Jive bridge requires CPU float64 tensors.")
        if self.stepwise:
            stress, self.trial_state = MaterialStep.apply(
                strain, self.state, self.native, self.history_step
            )
        else:
            history = (
                torch.stack(self.history)
                if self.history
                else strain.new_empty((0, *strain.shape))
            )
            stress = MaterialUpdate.apply(strain, history, self.native, self.history_step)
        self.trial = strain
        return stress

    def commit(self) -> None:
        if self.trial is not None:
            self.native.commit()
            if self.stepwise and self.trial_state is not None:
                self.state = self.trial_state
            elif self.training:
                self.history.append(self.trial)
            self.trial = None

    def cancel(self) -> None:
        self.native.cancel()
        self.trial = None


class _PythonPhaseContext:
    """
    Manages a single phase backed by a pure-Python `ConstitutiveModel` instance.
    Clones the prototype model per batch context so offline training and online Jive
    evaluation never overwrite each other's state tensors.
    """

    def __init__(
        self,
        model_prototype: ConstitutiveModel,
        total_points: int,
        components: int,
        training: bool = False,
    ):
        self.model = copy.deepcopy(model_prototype)
        self.model.configure(total_points=total_points, tensor_components=components)
        self.training = training
        self.has_trial = False

    def update(self, strain: Tensor) -> Tensor:
        stress = self.model.stress(strain)
        self.has_trial = True
        return stress

    def commit(self) -> None:
        if self.has_trial:
            self.model.commit(keep_graph=self.training)
            self.has_trial = False

    def cancel(self) -> None:
        if self.has_trial:
            self.model.cancel()
            self.has_trial = False


# ==============================================================================
# UNIFIED MULTI-PHASE MATERIAL LAYER
# ==============================================================================

MaterialSpec = Union[
    ConstitutiveModel,
    Callable[[int], Any],
    Sequence[Tuple[Union[ConstitutiveModel, Callable[[int], Any]], int]],
]


class MaterialLayer(nn.Module):
    """
    Composite-by-default physical constitutive layer for PRNNs.

    Orchestrates one or more constitutive material phases in parallel, partitioning
    the fictitious material points across phases and aggregating local Cauchy stresses.
    Seamlessly supports both pure-Python `ConstitutiveModel` instances and native
    Jem/Jive C++ `MaterialFactory` objects.

    Examples:
        # 1. Multi-phase composite (2 elastic points + 3 J2 plastic points):
        layer = MaterialLayer(
            materials=[
                (Elastic3DVectorized(young_modulus=400e3, poisson_ratio=0.2), 2),
                (J2Material3DVectorized(young_modulus=79.5e3, poisson_ratio=0.33), 3),
            ],
            components=6,
        )

        # 2. Multi-phase native Jive C++ materials:
        layer = MaterialLayer(
            materials=[(MaterialFactory(ELASTIC_CFG), 2), (MaterialFactory(J2_CFG), 3)],
            components=6,
        )

        # 3. Single-phase shorthand (infers n_matpts automatically):
        layer = MaterialLayer(materials=MaterialFactory(J2_CFG), components=6, n_matpts=5)
    """

    def __init__(
        self,
        materials: MaterialSpec,
        components: int,
        n_matpts: Optional[int] = None,
        history_step: float = 1.0e-7,
    ):
        """
        Initializes the multi-phase material layer specification.

        Args:
            materials: Either:
                - A list of `(material_or_factory, point_count)` tuples defining each phase, or
                - A single `ConstitutiveModel` or Jive `MaterialFactory` (requires `n_matpts`).
            components: Number of Voigt stress/strain components (e.g., 3 or 6).
            n_matpts: Total fictitious material points per sample. Required if `materials`
                is passed as a single un-tupled model/factory; if `materials` is a list of
                tuples, `n_matpts` is validated against the sum of phase point counts.
            history_step: Finite-difference perturbation step for Jive C++ history VJPs.
        """
        super().__init__()
        self.components = components
        self.history_step = history_step
        self.phases: List[Tuple[Union[ConstitutiveModel, Callable[[int], Any]], int]] = (
            self._normalize_spec(materials, n_matpts)
        )
        self.n_matpts = sum(count for _, count in self.phases)

        if self.n_matpts <= 0:
            raise ValueError("Total material points across all phases must be strictly positive.")

    @staticmethod
    def _normalize_spec(
        materials: MaterialSpec, n_matpts: Optional[int]
    ) -> List[Tuple[Union[ConstitutiveModel, Callable[[int], Any]], int]]:
        if isinstance(materials, (list, tuple)) and len(materials) > 0 and isinstance(materials[0], (list, tuple)):
            phases = [(mat, int(cnt)) for mat, cnt in materials]
            total = sum(cnt for _, cnt in phases)
            if n_matpts is not None and total != n_matpts:
                raise ValueError(
                    f"Sum of phase material points ({total}) does not match n_matpts ({n_matpts})."
                )
            return phases

        if n_matpts is None:
            raise ValueError(
                "When passing a single material/factory instead of a phase list, `n_matpts` must be specified."
            )
        return [(materials, int(n_matpts))]

    def create_context(self, batch_size: int, training: bool = False) -> "MaterialLayerContext":
        """
        Instantiates an stateful execution context for a batch of `batch_size` samples
        (used independently for offline training sequences and online Jive evaluation).
        """
        return MaterialLayerContext(
            phases=self.phases,
            batch_size=batch_size,
            components=self.components,
            training=training,
            history_step=self.history_step,
        )

    def describe_phases(self) -> List[dict]:
        """
        Returns serializable metadata describing each material phase (useful for checkpoints).
        """
        summary = []
        for idx, (mat, count) in enumerate(self.phases):
            info: dict[str, Any] = {"phase_index": idx, "n_points": count}
            if isinstance(mat, ConstitutiveModel):
                info["backend"] = "python"
                info["class"] = mat.__class__.__name__
                for attr in ("young_modulus", "poisson_ratio", "yield_spec"):
                    if hasattr(mat, attr):
                        info[attr] = getattr(mat, attr)
            else:
                info["backend"] = "jive_native"
                info["class"] = getattr(mat, "__name__", mat.__class__.__name__)
            summary.append(info)
        return summary


class MaterialLayerContext:
    """
    Stateful runtime execution context for a multi-phase MaterialLayer.
    Partitions strain tensors [batch_size * n_matpts, components] across phase contexts,
    evaluates stresses, and propagates commit/cancel operations.
    """

    def __init__(
        self,
        phases: List[Tuple[Union[ConstitutiveModel, Callable[[int], Any]], int]],
        batch_size: int,
        components: int,
        training: bool = False,
        history_step: float = 1.0e-7,
    ):
        self.batch_size = batch_size
        self.components = components
        self.n_matpts = sum(count for _, count in phases)
        self.phase_contexts: List[Tuple[Union[_PythonPhaseContext, _JivePhaseContext], int]] = []

        for mat, count in phases:
            total_phase_points = batch_size * count
            if isinstance(mat, ConstitutiveModel):
                ctx: Union[_PythonPhaseContext, _JivePhaseContext] = _PythonPhaseContext(
                    model_prototype=mat,
                    total_points=total_phase_points,
                    components=components,
                    training=training,
                )
            elif callable(mat):
                ctx = _JivePhaseContext(
                    factory=mat,
                    total_points=total_phase_points,
                    components=components,
                    training=training,
                    history_step=history_step,
                )
            else:
                raise TypeError(
                    f"Unsupported material phase specification: {type(mat)}. "
                    "Expected a ConstitutiveModel instance or a Jive MaterialFactory callable."
                )
            self.phase_contexts.append((ctx, count))

    def update(self, local_strain: Tensor) -> Tensor:
        """
        Evaluates trial local stresses across all material phases.

        Args:
            local_strain: Flattened strain tensor [batch_size * n_matpts, components].

        Returns:
            Flattened trial stress tensor [batch_size * n_matpts, components].
        """
        # Fast path for single-phase models (zero slicing/concatenation overhead)
        if len(self.phase_contexts) == 1:
            return self.phase_contexts[0][0].update(local_strain)

        strain_blocked = local_strain.reshape(self.batch_size, self.n_matpts, self.components)
        stress_blocks = []
        start = 0

        for ctx, count in self.phase_contexts:
            end = start + count
            sub_strain = strain_blocked[:, start:end, :].reshape(-1, self.components)
            sub_stress = ctx.update(sub_strain)
            stress_blocks.append(sub_stress.reshape(self.batch_size, count, self.components))
            start = end

        return torch.cat(stress_blocks, dim=1).reshape(-1, self.components)

    def commit(self) -> None:
        """Commits converged internal states across all material phases."""
        for ctx, _ in self.phase_contexts:
            ctx.commit()

    def cancel(self) -> None:
        """Reverts trial internal states across all material phases."""
        for ctx, _ in self.phase_contexts:
            ctx.cancel()