"""
Abstract base classes for Physically Recurrent Neural Networks (PRNN).
Follows PEP 8 styling and PEP 484 type hinting.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Material(nn.Module, ABC):
    """
    Abstract base class for physical constitutive models.
    """

    @abstractmethod
    def configure(self, total_points: int, tensor_components: int) -> None:
        """
        Allocates memory and initializes history tensors.

        Args:
            total_points (int): Batch size multiplied by the number of material points.
            tensor_components (int): Number of stress/strain components (e.g., 3 or 6).
        """
        pass

    @abstractmethod
    def update(self, micro_strain: torch.Tensor) -> torch.Tensor:
        """
        Computes local stresses from local strains using a vectorized return mapping.

        Args:
            micro_strain (torch.Tensor): Tensor of shape [total_points, tensor_components].

        Returns:
            torch.Tensor: Micro-stress tensor of shape [total_points, tensor_components].
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        Commits the converged internal history variables for the next time step.
        """
        pass

    @abstractmethod
    def get_history(self) -> torch.Tensor:
        """
        Retrieves internal variables (e.g., accumulated plasticity, damage).

        Returns:
            torch.Tensor: Internal history tensor.
        """
        pass


class PRNN(nn.Module, ABC):
    """
    Abstract base class for PRNN models.
    """

    @abstractmethod
    def forward(self, macro_strain_sequence: torch.Tensor) -> torch.Tensor:
        """
        Processes a full temporal sequence of macroscopic strains.

        Args:
            macro_strain_sequence (torch.Tensor): Shape [batch_size, seq_len, input_size].

        Returns:
            torch.Tensor: Predicted macro-stress sequence.
        """
        pass

    @abstractmethod
    def get_local_states(self, macro_strain_sequence: torch.Tensor) -> tuple:
        """
        Retrieves internal latent variables across the deformation path.
        """
        pass


class Encoder(nn.Module, ABC):
    """
    Abstract interface for kinematic de-homogenization layers (macro -> micro strain).
    """
    @abstractmethod
    def forward(self, macro_strain: torch.Tensor) -> torch.Tensor:
        """
        Maps macroscopic strain to flattened microscopic strain fields.
        Args:
            macro_strain (torch.Tensor): Shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Flattened micro-strains [batch_size, num_points * tensor_components].
        """
        pass

class Decoder(nn.Module, ABC):
    """
    Abstract base class for all multiscale homogenization layers (Decoders).
    Ensures that any implemented layer adheres to physical constraints and topological rules.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, scalar: float = 1.0) -> torch.Tensor:
        """
        Computes the final macroscopic stress.

        Args:
            x (Tensor): The micro-stress tensor. Shape [batch_size, latent_size].
            scalar (float, optional): Scaling factor. Defaults to 1.0.

        Returns:
            Tensor: The homogenized macroscopic stress.
        """
        pass

    @abstractmethod
    def get_latent(self, x: torch.Tensor, scalar: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the macroscopic stress alongside the local weighted latent fields.

        Args:
            x (Tensor): The micro-stress tensor. Shape [batch_size, latent_size].
            scalar (float, optional): Scaling factor. Defaults to 1.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the macroscopic stress and 
                                   the intermediate weighted latent values.
        """
        pass