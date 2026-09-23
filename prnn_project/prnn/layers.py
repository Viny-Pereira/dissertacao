"""
layers.py

Custom neural network layers for PRNN homogenization.
Enforces physical constraints, sparsity, and topological biases 
for micro-to-macro stress integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .interfaces import Homogenizer


class HomogenizerFactory:
    """
    Factory to instantiate homogenization layers based on a configuration string.
    """
    _registry = {
        "soft": lambda **k: SoftLayer(**k),
        "sparse_normalized": lambda **k: SparseNormalizedLayer(**k),
        "hyper": lambda **k: HyperLayer(**k),
        "abs_normalized": lambda **k: AbsNormalizedLayer(**k)
    }

    @classmethod
    def create(cls, layer_type: str, in_features: int, out_features: int, **kwargs) -> Homogenizer:
        """
        Instantiates and returns the requested homogenization layer.
        """
        layer_constructor = cls._registry.get(layer_type.lower())
        
        if layer_constructor is None:
            raise ValueError(f"Homogenizer '{layer_type}' not found. "
                             f"Available options: {list(cls._registry.keys())}")
            
        return layer_constructor(in_features=in_features, out_features=out_features, **kwargs)


class SoftLayer(Homogenizer):
    """
    Fully connected layer with strictly positive weights enforced via Softplus.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: torch.device = None, dtype: torch.dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if dtype is None:
            dtype = torch.float64

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
        positive_weights = self.softplus(self.weight)
        return F.linear(micro_stress, positive_weights * scalar, self.bias)

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Returns the macroscopic stress and the isolated local stress contributions 
        before the unit-partition summation.
        """
        positive_weights = self.softplus(self.weight) * scalar
        
        x_expanded = micro_stress.unsqueeze(1)  # [batch_size, 1, in_features]
        w_expanded = positive_weights.unsqueeze(0)  # [1, out_features, in_features]
        
        weighted_values = x_expanded * w_expanded  # [batch_size, out_features, in_features]
        output = weighted_values.sum(dim=2)
        
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values


class SparseNormalizedLayer(Homogenizer):
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
            w = torch.empty(self.num_subgroups, out_features, **factory_kwargs)
            nn.init.uniform_(w)
            w = torch.abs(w) / torch.abs(w).sum(dim=0, keepdim=True)
            
        self.weights = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        batch_size = micro_stress.size(0)
        
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        normalized_weights = scalar * torch.abs(self.weights) / torch.abs(self.weights).sum(dim=0, keepdim=True)

        weighted_values = x_reshaped * normalized_weights.unsqueeze(0)
        output = weighted_values.sum(dim=1)

        if self.bias is not None:
            output += self.bias

        return output

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)
        
        normalized_weights = scalar * torch.abs(self.weights) / torch.abs(self.weights).sum(dim=0, keepdim=True)
        weighted_values = x_reshaped * normalized_weights.unsqueeze(0)
        
        output = weighted_values.sum(dim=1)
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values.view(-1)


class HyperLayer(Homogenizer):
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
            
        self.weights = nn.Parameter(w)

    def forward(self, micro_stress: Tensor, scalar: float = 1.0) -> Tensor:
        batch_size = micro_stress.size(0)
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)

        normalized_weights = scalar * torch.abs(self.weights) / torch.abs(self.weights).sum(dim=0, keepdim=True)
        expanded_weights = normalized_weights.view(1, self.num_subgroups, 1)

        weighted_values = x_reshaped * expanded_weights
        output = weighted_values.sum(dim=1)

        return output

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        batch_size = micro_stress.size(0)
        x_reshaped = micro_stress.view(batch_size, self.num_subgroups, self.out_features)

        normalized_weights = scalar * torch.abs(self.weights) / torch.abs(self.weights).sum(dim=0, keepdim=True)
        expanded_weights = normalized_weights.view(1, self.num_subgroups, 1)

        weighted_values = x_reshaped * expanded_weights
        output = weighted_values.sum(dim=1)

        return output, weighted_values.view(-1)


class AbsNormalizedLayer(Homogenizer):
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
        normalized_weights = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        return F.linear(micro_stress, normalized_weights, self.bias)

    def get_latent(self, micro_stress: Tensor, scalar: float = 1.0) -> tuple[Tensor, Tensor]:
        normalized_weights = scalar * torch.abs(self.weight) / torch.abs(self.weight).sum(dim=1, keepdim=True)
        
        x_expanded = micro_stress.unsqueeze(1)  
        w_expanded = normalized_weights.unsqueeze(0)  
        
        weighted_values = x_expanded * w_expanded  
        output = weighted_values.sum(dim=2)
        
        if self.bias is not None:
            output += self.bias
            
        return output, weighted_values