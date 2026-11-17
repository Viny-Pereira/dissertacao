"""
Abstract base classes for Physically Recurrent Neural Networks (PRNN).
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class ConstitutiveModel(nn.Module, ABC):
    """
    Abstract base class for physical constitutive models.
    """

    @abstractmethod
    def configure(self, total_points: int, tensor_components: int = 6) -> None:
        """
        Allocates memory and initializes converged/trial history tensors.

        Args:
            total_points: Batch size multiplied by the number of material points.
            tensor_components: Number of stress/strain components (e.g., 3 or 6).
        """
        raise NotImplementedError

    @abstractmethod
    def stress(self, micro_strain: Tensor) -> Tensor:
        """
        Computes local stresses from local strains using a vectorized return mapping.

        Args:
            micro_strain (Tensor): Tensor of shape [total_points, tensor_components].

        Returns:
            Tensor: Micro-stress tensor of shape [total_points, tensor_components].
        """
        raise NotImplementedError

    @abstractmethod
    def commit(self) -> None:
        """
        Commits the converged internal history variables for the next time step.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """
        Reverts trial state variables back to the last committed state (used when
        Newton-Raphson diverges or during observe() queries).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_history(self) -> Tensor:
        """
        Retrieves internal variables (e.g., accumulated plasticity, damage).

        Returns:
            Tensor: Internal history tensor.
        """
        raise NotImplementedError



class Encoder(nn.Module, ABC):
    """
    Abstract interface for kinematic de-homogenization layers (macro -> micro strain).
    """

    @abstractmethod
    def forward(self, macro_strain: Tensor) -> Tensor:
        """
        Maps macroscopic strain to flattened microscopic strain fields.

        Args:
            macro_strain (Tensor): Shape [batch_size, input_dim].
        
        Returns:
            Tensor: Flattened micro-strains [batch_size, num_points * tensor_components].
        """
        raise NotImplementedError

class Decoder(nn.Module, ABC):
    """
    Abstract base class for all multiscale homogenization layers (Decoders).
    Ensures that any implemented layer adheres to physical constraints and topological rules.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes the homogenized macroscopic stress.

        Args:
            micro_stress: Micro-stress tensor of shape [batch_size, n_latents].
            scalar: Optional global scaling factor. Defaults to 1.0.
            e_raw: Optional localization tensor [n_matpts, n_features, n_features]
                required by thermodynamically consistent decoders.

        Returns:
            Homogenized macroscopic stress tensor of shape [batch_size, n_outputs].
        """
        raise NotImplementedError

    @abstractmethod
    def get_latent(
        self,
        micro_stress: Tensor,
        scalar: float = 1.0,
        e_raw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns the macroscopic stress alongside the isolated local weighted contributions.
        """
        raise NotImplementedError


class PRNN(nn.Module, ABC):
    """
    Abstract base class for PRNN models.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Offline training forward pass over batched strain sequences [batch, seq_len, n_features]."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, strain: Tensor) -> Tensor:
        """Online trial evaluation for a single step without committing history."""
        raise NotImplementedError

    @abstractmethod
    def commit(self) -> None:
        """Commits converged internal material state at the end of a load step."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """Reverts trial material state to the last committed step."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, strain: Tensor) -> Tensor:
        """Evaluates stress for post-processing/output without altering pending Newton trials."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets internal online material context for a new boundary value problem."""
        raise NotImplementedError

