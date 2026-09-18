"""
Concrete implementations of the PRNN architecture.
Separates the single-step physics (Cell) from the temporal unrolling (Sequence).
"""

import torch
import torch.nn as nn
from .interfaces import Material, PRNN, Homogenizer
from .layers import HomogenizerFactory


class PRNNCell(nn.Module):
    """
    Executes a single time step of the Physically Recurrent Neural Network.
    Maps macroscopic strain to microscopic strains, evaluates the physical material 
    model, and homogenizes the microscopic stresses back to macroscopic stresses.
    """

    def __init__(self, input_size: int, output_size: int, num_material_points: int, 
                 tensor_components: int, material_instance: Material, 
                 decoder_type: str = 'sparse_normalized'):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_material_points = num_material_points
        self.tensor_components = tensor_components
        self.latent_size = self.num_material_points * self.tensor_components
        
        # 1. Physics: The Material Model
        self.material = material_instance

        # 2. Encoder (Localization)
        # Maps macro-strain to latent micro-strains
        self.encoder = nn.Linear(in_features=self.input_size, 
                                 out_features=self.latent_size, 
                                 bias=False)
        
        # 3. Decoder (Homogenization)
        # Calls the Factory to instantiate the requested topology automatically
        self.decoder: Homogenizer = HomogenizerFactory.create(
            layer_type=decoder_type,
            in_features=self.latent_size,
            out_features=self.output_size,
            bias=False
        )

    def forward(self, macro_strain_step: torch.Tensor) -> torch.Tensor:
        """
        Computes the macroscopic stress for a single time increment.

        Args:
            macro_strain_step (torch.Tensor): Shape [batch_size, input_size].

        Returns:
            torch.Tensor: Macroscopic stress for the current step.
        """
        batch_size = macro_strain_step.size(0)
        total_points = batch_size * self.num_material_points

        # 1. Localization
        micro_strain_flat = self.encoder(macro_strain_step)
        
        # 2. Physics / Constitutive Evaluation
        micro_strain_reshaped = micro_strain_flat.view(total_points, self.tensor_components)
        micro_stress_reshaped, _ = self.material.update(micro_strain_reshaped)
        self.material.commit()
        
        # 3. Homogenization
        micro_stress_flat = micro_stress_reshaped.view(batch_size, self.latent_size)
        macro_stress_step = self.decoder(micro_stress_flat)
        
        return macro_stress_step


class PRNNSequence(PRNN):
    """
    Handles the temporal sequence unrolling for the PRNNCell.
    Processes full loading paths incrementally.
    """

    def __init__(self, cell: PRNNCell):
        super().__init__()
        self.cell = cell

    def forward(self, macro_strain_sequence: torch.Tensor) -> torch.Tensor:
        """
        Iterates over the sequence length and collects the stress outputs.

        Args:
            macro_strain_sequence (torch.Tensor): Shape [batch_size, seq_len, input_size].

        Returns:
            torch.Tensor: Macro-stress tensor of shape [batch_size, seq_len, output_size].
        """
        batch_size, seq_len, _ = macro_strain_sequence.size()
        
        # Configure material memory before the temporal loop begins
        total_points = batch_size * self.cell.num_material_points
        self.cell.material.configure(total_points, self.cell.tensor_components)
        
        # Pre-allocate output tensor on the same device as the input
        output_stress_sequence = torch.zeros(
            (batch_size, seq_len, self.cell.output_size), 
            device=macro_strain_sequence.device, 
            dtype=macro_strain_sequence.dtype
        )
        
        for t in range(seq_len):
            current_strain = macro_strain_sequence[:, t, :]
            current_stress = self.cell(current_strain)
            output_stress_sequence[:, t, :] = current_stress

        return output_stress_sequence

    def get_local_states(self, macro_strain_sequence: torch.Tensor):
        """
        Executa a sequência extraindo as variáveis de histórico e tensores latentes 
        para plotagem e validação (análogo ao antigo getOutputMatPts).
        """
        batch_size, seq_len, num_components = macro_strain_sequence.size()
        
        # Pré-aloca tensores para guardar o histórico
        macro_stress_seq = torch.zeros_like(macro_strain_sequence)
        
        # Pega os tamanhos corretos baseados na PRNNCell
        latent_size = self.cell.latent_size
        num_points = self.cell.num_material_points
        
        local_strains_seq = torch.zeros((batch_size, seq_len, latent_size), device=macro_strain_sequence.device)
        local_stresses_seq = torch.zeros((batch_size, seq_len, latent_size), device=macro_strain_sequence.device)
        local_history_seq = torch.zeros((batch_size, seq_len, num_points), device=macro_strain_sequence.device)
        
        # Configura o material (zera o histórico)
        self.cell.material.configure(total_points=batch_size * num_points, 
                                     tensor_components=num_components)
        
        for t in range(seq_len):
            current_strain = macro_strain_sequence[:, t, :]
            
            # 1. Encoder manual (De-homogeneização)
            micro_strains = self.cell.encoder(current_strain)
            local_strains_seq[:, t, :] = micro_strains
            
            # 2. Física manual
            micro_strains_reshaped = micro_strains.view(batch_size * num_points, num_components)
            micro_stresses_reshaped, _ = self.cell.material.update(micro_strains_reshaped)
            self.cell.material.commit()
            
            # Salva histórico do material e micro-tensões
            local_history_seq[:, t, :] = self.cell.material.get_history().view(batch_size, num_points)
            
            micro_stresses = micro_stresses_reshaped.view(batch_size, latent_size)
            local_stresses_seq[:, t, :] = micro_stresses
            
            # 3. Decoder manual (Homogeneização)
            macro_stress_seq[:, t, :] = self.cell.decoder(micro_stresses)
            
        return macro_stress_seq, local_strains_seq, local_stresses_seq, local_history_seq