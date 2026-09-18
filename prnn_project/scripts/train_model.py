"""
Execution script for training the Physically Recurrent Neural Network (PRNN)
on structural stress-strain path datasets.
"""

import torch
import numpy as np
import random
import os

# Import modules from our custom 'prnn' package
from prnn.materials import J2Material3DVectorized
from prnn.models import PRNNCell, PRNNSequence
from prnn.utils import StressStrainDataset, train_and_save_model, plot_convergence_from_csv


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Hyperparameters and Global Configuration
    # -------------------------------------------------------------------------
    input_size = 6
    output_size = 6
    num_material_points = 5
    tensor_components = 6
    
    sequence_length = 61
    batch_size = 10
    epochs = 1000
    learning_rate = 1e-3
    patience = 50
    random_seed = 42

    # Reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_default_dtype(torch.float64)

    # Device configuration (Force CPU or switch to CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_path = "data/monotonic_loading.out"  # Update with your actual data file path
    weight_path = "trained_models/prnn_composite_loading_5_sparse.pth"
    csv_path = "trained_models/historico_5_sparse.csv"

    # -------------------------------------------------------------------------
    # 2. Dataset and Dataloader Setup
    # -------------------------------------------------------------------------
    print("Initializing datasets...")
    
    # Assuming features are columns 0-5 and targets are columns 6-11
    feature_indices = list(range(0, 6))
    target_indices = list(range(6, 12))

    full_dataset = StressStrainDataset(
        filename=data_path,
        features=feature_indices,
        targets=target_indices,
        seq_length=sequence_length,
        normalize_features=True
    )

    # Split into training and validation sets (e.g., 80% train, 20% val)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total sequences: {total_samples} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Instantiation
    # -------------------------------------------------------------------------
    print("Constructing the PRNN architecture...")

    # Instantiate the physical material model (Vectorized J2 Plasticity)
    material_model = J2Material3DVectorized(device=device)

    # Instantiate the single-step cell with a sparse decoder topology
    cell = PRNNCell(
        input_size=input_size,
        output_size=output_size,
        num_material_points=num_material_points,
        tensor_components=tensor_components,
        material_instance=material_model,
        decoder_type='sparse'
    )

    # Wrap the cell into the sequence unroller
    prnn_model = PRNNSequence(cell=cell).to(device)

    # -------------------------------------------------------------------------
    # 4. Training Execution
    # -------------------------------------------------------------------------
    train_and_save_model(
        model=prnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        weight_path=weight_path,
        csv_path=csv_path,
        lr=learning_rate,
        epochs=epochs,
        patience=patience
    )

    # -------------------------------------------------------------------------
    # 5. Post-Training Visualization
    # -------------------------------------------------------------------------
    print("Plotting convergence history...")
    plot_convergence_from_csv(csv_path, ignore_initial=10)


if __name__ == "__main__":
    main()