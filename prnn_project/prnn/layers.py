"""
layers.py

Custom neural network layers for PRNN homogenization.
Enforces physical constraints, sparsity, and topological biases 
for micro-to-macro stress integration.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .interfaces import Decoder, Encoder

class LinearEncoder(Encoder):
    """
    Unconstrained linear de-homogenization layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, macro_strain: torch.Tensor) -> torch.Tensor:
        return self.linear(macro_strain)

'''
class ConsistentEncoder(Encoder):
    """
    Kinematically consistent de-homogenization layer enforcing strict partition 
    of unity: sum_k(w_k * A_k) = Identity, ruling out unphysical volumetric modes.
    """
    def __init__(self, num_material_points: int, tensor_components: int = 6,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.num_points = num_material_points
        self.dim = tensor_components
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Unconstrained learnable perturbation around identity
        self.raw_weights = nn.Parameter(
            torch.zeros((self.num_points, self.dim, self.dim), device=self.device, dtype=self.dtype)
        )
        nn.init.normal_(self.raw_weights, mean=0.0, std=1e-3)

        self.register_buffer(
            "identity", 
            torch.eye(self.dim, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.num_points, 1, 1)
        )

    def forward(self, macro_strain: torch.Tensor) -> torch.Tensor:
        batch_size = macro_strain.size(0)
        
        # Enforce zero-mean perturbation across material points
        perturbation_mean = torch.mean(self.raw_weights, dim=0, keepdim=True)
        zero_mean_perturbation = self.raw_weights - perturbation_mean
        localization_tensors = self.identity + zero_mean_perturbation  # [Np, 6, 6]

        macro_expanded = macro_strain.unsqueeze(1).unsqueeze(-1)         # [batch, 1, 6, 1]
        a_expanded = localization_tensors.unsqueeze(0)                   # [1, Np, 6, 6]
        
        micro_strain = torch.matmul(a_expanded, macro_expanded).squeeze(-1) # [batch, Np, 6]
        return micro_strain.view(batch_size, self.num_points * self.dim)
'''

class EncoderFactory:
    """
    Factory to instantiate kinematic de-homogenization layers via class registry.
    """
    _registry = {
        "linear": LinearEncoder,
        #"consistent": ConsistentEncoder,
        #"kinematic": ConsistentEncoder
    }

    @classmethod
    def create(cls,
        encoder_type: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        **kwargs) -> Encoder:
        constructor = cls._registry.get(encoder_type.lower())
        if constructor is None:
            raise ValueError(f"Encoder '{encoder_type}' not found. Available: {list(cls._registry.keys())}")
        return constructor(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            **kwargs
        )


class SoftLayer(Decoder):
    """
    Fully connected layer with strictly positive weights enforced via Softplus.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: torch.device = None, dtype: torch.dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype or torch.float64}
        super().__init__()


        self.in_features = in_features
        self.out_features = out_features
        self.softplus = nn.Softplus()
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        positive_weight = self.softplus(self.weight)
        return F.linear(micro_stress, positive_weight * scalar, self.bias)

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Returns the macroscopic stress and the isolated local stress contributions 
        before the unit-partition summation.
        """
        positive_weight = self.softplus(self.weight) * scalar
        
        x_expanded = micro_stress.unsqueeze(1)  # [batch_size, 1, in_features]
        w_expanded = positive_weight.unsqueeze(0)  # [1, out_features, in_features]
        
        weighted_values = x_expanded * w_expanded  # [batch_size, out_features, in_features]
        output = weighted_values.sum(dim=2)
        
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values

class SparseLayer(Decoder):
    """
    Sparse homogenization layer WITHOUT partition of unity normalization (unconstrained positive weights).
    Restricts cross-connections by linking each macroscopic stress component strictly to its 
    corresponding microscopic stress components, without enforcing the sum-to-one constraint.
    
    Weight tensor shape: [out_features, num_subgroups] (e.g., [6, N_points]).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        factory_kwargs = {'device': device, 'dtype': dtype or torch.float64}
        super().__init__()

        self.out_features = out_features
        self.num_subgroups = in_features // out_features

        # Initialize weight tensor with shape [out_features, num_subgroups]
        with torch.no_grad():
            w = torch.empty(self.out_features, self.num_subgroups, **factory_kwargs)
            nn.init.uniform_(w, a=0.1, b=1.0)
            
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        batch_size = micro_stress.size(0)
        
        # Reshape ensuring physical continuity: [batch_size, num_subgroups, out_features]
        micro_stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        
        # Free positive weights (no L1 normalization across points)
        effective_weight = scalar * torch.abs(self.weight)
        
        # Transpose from [out_features, num_subgroups] to [num_subgroups, out_features]
        # and expand to [1, num_subgroups, out_features] for proper broadcasting
        weights_aligned = effective_weight.t().unsqueeze(0)
        
        # Element-wise weighting: [batch, num_subgroups, out_features] * [1, num_subgroups, out_features]
        weighted_values = micro_stress_reshaped * weights_aligned
        
        # Homogenization: sum across material points (dim=1) -> [batch_size, out_features]
        output = weighted_values.sum(dim=1)

        if self.bias is not None:
            output += self.bias

        return output

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        micro_stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        
        effective_weight = scalar * torch.abs(self.weight)
        weights_aligned = effective_weight.t().unsqueeze(0)
        
        weighted_values = micro_stress_reshaped * weights_aligned
        output = weighted_values.sum(dim=1)
        
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values.view(batch_size, -1)

    
class SparseNormalizedLayer(Decoder):
    """
    Sparse homogenization layer. Restricts cross-connections by linking each 
    macroscopic stress component strictly to its corresponding microscopic components.
    Weights are normalized to sum to 1.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: torch.device = None, dtype: torch.dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if dtype is None:
            dtype = torch.float64

        self.out_features = out_features
        self.num_subgroups = in_features // out_features

        with torch.no_grad():
            w = torch.empty(self.out_features, self.num_subgroups, **factory_kwargs)
            nn.init.uniform_(w)
            w = torch.abs(w) / torch.abs(w).sum(dim=1, keepdim=True)
            
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        batch_size = micro_stress.size(0)
        
        micro_stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)

        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        
        weights_aligned = normalized_weight.t().unsqueeze(0)
        
        weighted_values = micro_stress_reshaped * weights_aligned
        output = weighted_values.sum(dim=1)

        if self.bias is not None:
            output += self.bias

        return output

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        micro_stress_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        
        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        weights_aligned = normalized_weight.t().unsqueeze(0)
                
        weighted_values = micro_stress_reshaped * weights_aligned
        
        output = weighted_values.sum(dim=1)
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values.view(-1)


class HyperLayer(Decoder):
    """
    Assigns a single scalar weight to all stress components of a given material point.
    Weights are normalized across all material points to sum to 1.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: torch.device = None, dtype: torch.dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if dtype is None:
            dtype = torch.float64

        self.out_features = out_features
        self.num_subgroups = in_features // out_features

        with torch.no_grad():
            w = torch.empty(self.num_subgroups, **factory_kwargs)
            nn.init.uniform_(w)
            w = torch.abs(w) / torch.abs(w).sum(dim=0, keepdim=True)
            
        self.weight = nn.Parameter(w)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        batch_size = micro_stress.size(0)
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)

        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=0, keepdim=True)
        expanded_weight = normalized_weight.view(1, self.num_subgroups, 1)

        weighted_values = x_reshaped * expanded_weight
        output = weighted_values.sum(dim=1)

        return output

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)

        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=0, keepdim=True)
        expanded_weight = normalized_weight.view(1, self.num_subgroups, 1)

        weighted_values = x_reshaped * expanded_weight
        output = weighted_values.sum(dim=1)

        return output, weighted_values.view(-1)


class AbsNormalizedLayer(Decoder):
    """
    Fully connected layer where weights are enforced as absolute values 
    and normalized by row (dim=1) so that the contributions to each output sum to 1.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: torch.device = None, dtype: torch.dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if dtype is None:
            dtype = torch.float64

        self.in_features = in_features
        self.out_features = out_features
        self.num_subgroups = in_features // out_features

        with torch.no_grad():
            w = torch.empty((out_features, in_features), **factory_kwargs)
            nn.init.uniform_(w)
            w = torch.abs(w) / torch.abs(w).sum(dim=1, keepdim=True)

        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        return F.linear(micro_stress, normalized_weight, self.bias)

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        normalized_weight = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        
        x_expanded = micro_stress.unsqueeze(1)  
        w_expanded = normalized_weight.unsqueeze(0)  
        
        weighted_values = x_expanded * w_expanded  
        output = weighted_values.sum(dim=2)
        
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values

class DecoderFactory:
    """
    Factory to instantiate homogenization layers based on a configuration string.
    """
    _registry = {
        "dense": SoftLayer,
        "sparse": SparseLayer,                     
        "sparse_normalized":SparseNormalizedLayer,
        "hyper":HyperLayer,
        "abs_normalized":AbsNormalizedLayer
    }

    @classmethod
    def create(cls, 
        decoder_type: str,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        **kwargs) -> Decoder:
        """
        Instantiates and returns the requested homogenization layer.
        """
        constructor = cls._registry.get(decoder_type.lower())
        
        if constructor is None:
            raise ValueError(f"Homogenizer '{decoder_type}' not found. "
                             f"Available options: {list(cls._registry.keys())}")
            
        return constructor(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            **kwargs
        )

