"""
Custom neural network layers for PRNN localization (Encoders) and homogenization (Decoders).
Enforces physical constraints, structural sparsity, Hill-Mandel thermodynamic consistency,
and unified weight normalization.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .interfaces import Decoder, Encoder


def _positive_activation(weight: Tensor, positive_fn: str) -> Tensor:
    """Applies element-wise non-negativity transform ('abs' or 'softplus')."""
    if positive_fn == "abs":
        return torch.abs(weight)
    if positive_fn == "softplus":
        return F.softplus(weight)
    raise ValueError(f"Unsupported positive_fn '{positive_fn}'. Choose 'abs' or 'softplus'.")


# ==============================================================================
# ENCODERS (Localization: macro_strain -> micro_strain)
# ==============================================================================

class LinearEncoder(Encoder):
    """
    Unconstrained linear localization layer: eps_micro = W * eps_macro.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype or torch.float64,
        )

    def forward(self, macro_strain: Tensor) -> Tensor:
        return self.linear(macro_strain)


class ConsistentEncoder(Encoder):
    r"""
    Kinematic localization layer storing per-point deformation concentration tensors E_j:
        eps_j = E_j * eps_macro
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if out_features % in_features != 0:
            raise ValueError("out_features must be a multiple of in_features")

        self.dim = in_features
        self.num_points = out_features // in_features
        factory_kwargs = {"device": device, "dtype": dtype or torch.float64}

        # Initialize each E_j around the identity tensor I_dim + small perturbation
        e_init = torch.eye(self.dim, **factory_kwargs).unsqueeze(0).repeat(self.num_points, 1, 1)
        perturbation = torch.randn((self.num_points, self.dim, self.dim), **factory_kwargs) * 1e-3
        self.E_raw = nn.Parameter(e_init + perturbation)

    def forward(self, macro_strain: Tensor) -> Tensor:
        batch_size = macro_strain.size(0)
        macro_expanded = macro_strain.unsqueeze(1).unsqueeze(-1)  # [batch, 1, dim, 1]
        e_expanded = self.E_raw.unsqueeze(0)                      # [1, Np, dim, dim]
        micro_strain = torch.matmul(e_expanded, macro_expanded).squeeze(-1)  # [batch, Np, dim]
        return micro_strain.reshape(batch_size, self.num_points * self.dim)


class EncoderFactory:
    """
    Factory to instantiate localization layers via class registry.
    """

    _registry = {
        "linear": LinearEncoder,
        "consistent": ConsistentEncoder,
        "tc": ConsistentEncoder,
    }

    @classmethod
    def create(
        cls,
        encoder_type: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Encoder:
        constructor = cls._registry.get(encoder_type.lower())
        if constructor is None:
            raise ValueError(
                f"Encoder '{encoder_type}' not found. Available: {list(cls._registry.keys())}"
            )
        return constructor(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype or torch.float64,
            **kwargs,
        )


# ==============================================================================
# DECODERS (Homogenization: micro_stress -> macro_stress)
# ==============================================================================

class DenseLayer(Decoder):
    """
    Fully connected homogenization layer with positive weights (via 'abs' or 'softplus')
    and optional row-wise partition-of-unity normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize_weights: bool = False,
        positive_fn: str = "softplus",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype or torch.float64}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.normalize_weights = normalize_weights
        self.positive_fn = positive_fn

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weights(self) -> Tensor:
        pos_w = _positive_activation(self.weight, self.positive_fn)
        if self.normalize_weights:
            return pos_w / (pos_w.sum(dim=1, keepdim=True) + 1e-12)
        return pos_w

    def forward(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> Tensor:
        w = scalar * self.effective_weights()
        return F.linear(micro_stress, w, self.bias)

    def get_latent(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        w = scalar * self.effective_weights()
        weighted_values = micro_stress.unsqueeze(1) * w.unsqueeze(0)  # [batch, out, in]
        output = weighted_values.sum(dim=2)
        if self.bias is not None:
            output = output + self.bias
        return output, weighted_values.reshape(batch_size, -1)


class SparseLayer(Decoder):
    """
    Component-wise sparse homogenization layer linking each macroscopic stress component
    strictly to its corresponding microscopic stress components across material points.
    Weight tensor shape: [out_features, n_matpts].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize_weights: bool = False,
        positive_fn: str = "abs",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype or torch.float64}
        super().__init__()

        self.out_features = out_features
        self.num_subgroups = in_features // out_features
        self.normalize_weights = normalize_weights
        self.positive_fn = positive_fn

        with torch.no_grad():
            w = torch.empty((self.out_features, self.num_subgroups), **factory_kwargs)
            nn.init.uniform_(w, a=0.1, b=1.0)
            if self.normalize_weights:
                pos_w = _positive_activation(w, self.positive_fn)
                w = pos_w / pos_w.sum(dim=1, keepdim=True)

        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def effective_weights(self) -> Tensor:
        pos_w = _positive_activation(self.weight, self.positive_fn)
        if self.normalize_weights:
            return pos_w / (pos_w.sum(dim=1, keepdim=True) + 1e-12)
        return pos_w

    def forward(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = micro_stress.size(0)
        stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        weights_aligned = (scalar * self.effective_weights()).t().unsqueeze(0)  # [1, Np, out]
        output = (stress_reshaped * weights_aligned).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_latent(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        weights_aligned = (scalar * self.effective_weights()).t().unsqueeze(0)
        weighted_values = stress_reshaped * weights_aligned
        output = weighted_values.sum(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output, weighted_values.reshape(batch_size, -1)


class PointLayer(Decoder):
    """
    Point-wise scalar homogenization layer assigning a single scalar weight w_p
    to all stress components of each fictitious material point p.
    Weight tensor shape: [n_matpts, 1].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize_weights: bool = False,
        positive_fn: str = "abs",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype or torch.float64}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_subgroups = in_features // out_features
        self.normalize_weights = normalize_weights
        self.positive_fn = positive_fn

        with torch.no_grad():
            if self.normalize_weights:
                w = torch.ones((self.num_subgroups, 1), **factory_kwargs)
            else:
                w = torch.full((self.num_subgroups, 1), 1.0 / self.num_subgroups, **factory_kwargs)

        self.raw_weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def effective_weights(self) -> Tensor:
        pos_w = _positive_activation(self.raw_weight, self.positive_fn)
        if self.normalize_weights:
            return pos_w / (pos_w.sum(dim=0, keepdim=True) + 1e-12)
        return pos_w

    def forward(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = micro_stress.size(0)
        stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        weights_aligned = (scalar * self.effective_weights()).unsqueeze(0)  # [1, Np, 1]
        output = (stress_reshaped * weights_aligned).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_latent(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        weights_aligned = (scalar * self.effective_weights()).unsqueeze(0)
        weighted_stress = stress_reshaped * weights_aligned
        output = weighted_stress.sum(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output, weighted_stress.reshape(batch_size, -1)


class ConsistentDecoder(Decoder):
    r"""
    Thermodynamically consistent homogenization layer satisfying Hill-Mandel:
        sigma_macro = sum_j ( w_j * E_j^T * sigma_j )
    Receives E_raw explicitly during forward() to keep module ownership decoupled
    for TorchScript and checkpoint serialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize_weights: bool = True,
        positive_fn: str = "softplus",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype or torch.float64}
        super().__init__()

        self.dim = out_features
        self.num_points = in_features // out_features
        self.normalize_weights = normalize_weights
        self.positive_fn = positive_fn

        w_init = torch.full((self.num_points, 1), 1.0 / self.num_points, **factory_kwargs)
        self.w_raw = nn.Parameter(w_init)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def volume_weights(self) -> Tensor:
        pos_w = _positive_activation(self.w_raw, self.positive_fn)
        if self.normalize_weights:
            return pos_w / (pos_w.sum(dim=0, keepdim=True) + 1e-12)
        return pos_w

    def forward(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> Tensor:
        if e_raw is None:
            raise ValueError("ConsistentDecoder requires 'e_raw' from ConsistentEncoder in forward()")

        batch_size = micro_stress.size(0)
        stress_blocked = micro_stress.view(batch_size, self.num_points, self.dim).unsqueeze(-1)
        e_t = e_raw.transpose(-2, -1).unsqueeze(0)  # [1, Np, dim, dim]
        pulled_stress = torch.matmul(e_t, stress_blocked).squeeze(-1)  # [batch, Np, dim]

        w = (scalar * self.volume_weights()).unsqueeze(0)  # [1, Np, 1]
        macro_stress = (pulled_stress * w).sum(dim=1)

        if self.bias is not None:
            macro_stress = macro_stress + self.bias
        return macro_stress

    def get_latent(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if e_raw is None:
            raise ValueError("ConsistentDecoder requires 'e_raw' from ConsistentEncoder in get_latent()")

        batch_size = micro_stress.size(0)
        stress_blocked = micro_stress.view(batch_size, self.num_points, self.dim).unsqueeze(-1)
        e_t = e_raw.transpose(-2, -1).unsqueeze(0)
        pulled_stress = torch.matmul(e_t, stress_blocked).squeeze(-1)

        w = (scalar * self.volume_weights()).unsqueeze(0)
        weighted_values = pulled_stress * w
        macro_stress = weighted_values.sum(dim=1)

        if self.bias is not None:
            macro_stress = macro_stress + self.bias
        return macro_stress, weighted_values.reshape(batch_size, -1)


class DecoderFactory:
    """
    Factory to instantiate homogenization layers based on configuration strings.
    Supports unified classes with `normalize_weights` flag as well as backward-compatible
    aliases (e.g., 'sparse_normalized', 'point_normalized', 'hyper').
    """

    _registry = {
        "dense": (DenseLayer, {}),
        "soft": (DenseLayer, {"positive_fn": "softplus"}),
        "dense_normalized": (DenseLayer, {"normalize_weights": True, "positive_fn": "softplus"}),
        "abs_normalized": (DenseLayer, {"normalize_weights": True, "positive_fn": "abs"}),
        "sparse": (SparseLayer, {}),
        "sparse_normalized": (SparseLayer, {"normalize_weights": True}),
        "point": (PointLayer, {}),
        "point_normalized": (PointLayer, {"normalize_weights": True}),
        "hyper": (PointLayer, {"normalize_weights": True}),
        "consistent": (ConsistentDecoder, {}),
        "tc": (ConsistentDecoder, {}),
    }

    @classmethod
    def create(
        cls,
        decoder_type: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Decoder:
        entry = cls._registry.get(decoder_type.lower())
        if entry is None:
            raise ValueError(
                f"Decoder '{decoder_type}' not found. Available options: {list(cls._registry.keys())}"
            )

        constructor, default_kwargs = entry
        merged_kwargs = {**default_kwargs, **kwargs}
        return constructor(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype or torch.float64,
            **merged_kwargs,
        )