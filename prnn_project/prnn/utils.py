"""
Utility classes and functions for data processing, training, 
and evaluating PRNN models.
"""

import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any, Optional, List, Tuple

from .models import PRNNCell


class Normalizer:
    """
    Normalization for strain features.
    Scales strain data to a [-1, 1] interval.
    """
    def __init__(self, x_tensor: torch.Tensor):
        self.min = x_tensor.min(dim=0).values
        self.max = x_tensor.max(dim=0).values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Min-Max scaling to map features to [-1, 1]."""
        return 2.0 * ((x - self.min) / (self.max - self.min)) - 1.0


class StressStrainDataset(Dataset):
    """
    Custom dataset for handling stress-strain paths.
    Data is loaded with pandas, and stress-strain pairs are split 
    into paths of 'seq_length' time steps. 
    """
    def __init__(self, filename: str, features: List[int], targets: List[int], 
                 seq_length: int, normalize_features: bool = False, 
                 normalizer: Optional[Normalizer] = None, dtype: torch.dtype = torch.float64):
        
        df = pd.read_csv(filename, sep=r"\s+", header=None)
        self.seq_length = seq_length

        self.X = torch.tensor(df[features].values, dtype=dtype)
        self.T = torch.tensor(df[targets].values, dtype=dtype)

        self.normalize_features = normalize_features

        if self.normalize_features:
            self._normalizer = normalizer if normalizer else Normalizer(self.X)

    def __len__(self) -> int:
        """Returns the total number of complete load paths (sequences) in the dataset."""
        return int(self.X.shape[0] / self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts a full temporal load path for the given index.
        Returns a tuple of (strain_path, stress_path).
        """
        start = idx * self.seq_length
        end = start + self.seq_length

        strain_path = self.X[start:end, :]
        stress_path = self.T[start:end, :]

        if self.normalize_features:
            return self._normalizer(strain_path), stress_path
        else:
            return strain_path, stress_path

    def get_normalizer(self) -> Optional[Normalizer]:
        """Returns the normalizer instance if feature normalization is enabled."""
        return self._normalizer if self.normalize_features else None


class RelativeError(nn.Module):
    """
    Custom loss function computing the mean relative error.
    Includes a small epsilon barrier to prevent division by zero.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        numerator = torch.norm(y_pred - y_true, dim=-1)
        denominator = torch.clamp(torch.norm(y_true, dim=-1), min=1.0)
        error = numerator / denominator
        return torch.mean(error)
    

class RobustRelativeError(nn.Module):
    """
    Custom loss function combining absolute and relative error properties.
    Prevents instability by regularizing the denominator with a reference stress.
    """
    def __init__(self, sigma_ref: float = 1.0):
        super().__init__()
        self.sigma_ref = sigma_ref

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = torch.norm(y_true - y_pred, p=2, dim=-1)
        # Regularized denominator: 
        # Behaves as absolute error when ||y_true|| is small, relative when large.
        denominator = torch.norm(y_true, p=2, dim=-1) + self.sigma_ref
        return torch.mean(diff / denominator)


class Trainer:
    """
    Class for handling standard network training tasks.
    Wraps an existing PyTorch model and performs training and evaluation.
    Early stopping is implemented with adjustable patience.
    """
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                 loss: Optional[nn.Module] = None, scheduler: Optional[Any] = None):
        self._model = model
        self._epoch = 0
        self._criterion = loss if loss else nn.MSELoss()
        self._optimizer = optimizer if optimizer else torch.optim.Adam(self._model.parameters())
        self.device = next(self._model.parameters()).device
        self.scheduler = scheduler
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        self._best_val = float('inf')
        self._best_state_dict = copy.deepcopy(self._model.state_dict())

        total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"Total trainable parameter count: {total_params}\n")

    def train(self, training_loader: torch.utils.data.DataLoader, 
              validation_loader: torch.utils.data.DataLoader, 
              epochs: int = 100, patience: int = 20, 
              interval: int = 1, verbose: bool = True) -> None:
        """
        Executes the main training loop with periodic validation and early stopping.

        This method handles device placement (CPU/GPU), performs forward and backward 
        passes, applies gradient clipping to prevent exploding gradients in recurrent 
        paths, and tracks the best model state based on validation loss.

        Args:
            training_loader (DataLoader): DataLoader providing batches of strain and stress paths for training.
            validation_loader (DataLoader): DataLoader providing batches for validation.
            epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            patience (int, optional): Number of validation intervals to wait for an improvement 
                in validation loss before stopping the training. Defaults to 20.
            interval (int, optional): Frequency (in epochs) at which the validation set 
                is evaluated. Defaults to 1.
            verbose (bool, optional): If True, prints loss progression and early stopping 
                messages to the console. Defaults to True.
        """
        self._model.train()
        stall_iters = 0
        torch.autograd.set_detect_anomaly(False)

        for i in range(epochs):
            # Acumuladores ponderados para treino
            running_loss_total = 0.0
            total_train_samples = 0

            # Domain-driven names: inputs (strain_seq) and targets (stress_seq)
            for inputs, targets in training_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                predictions = self._model(inputs)
                loss = self._criterion(predictions, targets)
                
                self._optimizer.zero_grad(spet_to_none=True)
                loss.backward()
                
                # Aggressive gradient clipping to prevent exploding gradients in PRNN unrolling
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.1)
                
                self._optimizer.step()
                
                current_batch_size = inputs.size(0)
                running_loss_total += loss.item() * current_batch_size
                total_train_samples += current_batch_size

            epoch_train_loss = running_loss_total / total_train_samples
            self._epoch += 1

            if i < interval or i % interval == 0:
                with torch.no_grad():
                    self._model.eval()
                    running_val_loss_total = 0.0
                    total_val_samples = 0
                    
                    for x, t in validation_loader:
                        x = x.to(self.device, non_blocking=True)
                        t = t.to(self.device, non_blocking=True)
                        y = self._model(x)
                        loss = self._criterion(y, t)
                        
                        current_batch_size = x.size(0)
                        running_val_loss_total += loss.item() * current_batch_size
                        total_val_samples += current_batch_size
                        
                    epoch_val_loss = running_val_loss_total / total_val_samples
                    
                    self.train_losses.append(epoch_train_loss)
                    self.val_losses.append(epoch_val_loss)
                    self._model.train()
                    
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_val_loss)
                    else:
                        self.scheduler.step()

                if verbose:
                    print(f"Epoch {self._epoch} | Training Loss: {epoch_train_loss:.6e} | Validation Loss: {epoch_val_loss:.6e}")

                if epoch_val_loss <= self._best_val:
                    self._best_val = epoch_val_loss
                    self._best_state_dict = copy.deepcopy(self._model.state_dict())
                    stall_iters = 0

                    if verbose:
                        print("The best historical model has been updated. Resetting early stop counter.")
                else:
                    stall_iters += interval if i > interval else 1

                if stall_iters >= patience:
                    if verbose:
                        print("Early stopping criterion reached.")
                    break

        print("End of training.")

    def eval(self, test_loader: torch.utils.data.DataLoader, verbose: bool = True) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Evaluates the model on a test/validation loader using the best historical weights.
        
        Returns:
            Tuple[float, float, torch.Tensor, torch.Tensor]:
                - mse_loss: Mean Squared Error across all sequences.
                - rmse_loss: Root Mean Squared Error (in MPa).
                - all_predictions: Concatenated predicted stresses [N_samples, seq_len, 6].
                - all_targets: Concatenated ground-truth stresses [N_samples, seq_len, 6].
        """
        # Restaura temporariamente os melhores pesos salvos durante o treino
        
        live_state = copy.deepcopy(self._model.state_dict())
        self._model.load_state_dict(self._best_state_dict)
        self._model.eval()

        running_loss_total = 0.0
        total_samples = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                predicted_stress = self._model(inputs)
                loss = self._criterion(predicted_stress, targets)

                current_batch_size = inputs.size(0)
                running_loss_total += loss.item() * current_batch_size
                total_samples += current_batch_size

                predictions_list.append(predicted_stress.detach().cpu())
                targets_list.append(targets.detach().cpu())

        # Restaura o estado original de treino
        self._model.load_state_dict(live_state)
        self._model.train()

        mse_loss = running_loss_total / total_samples
        rmse_loss = np.sqrt(mse_loss)
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        if verbose:
            print(f"\n{'='*50}")
            print(f"EVALUATION RESULTS (Best Model Checkpoint):")
            print(f"  Total Sequences Evaluated : {total_samples}")
            print(f"  Aggregated MSE            : {mse_loss:.6e} MPa²")
            print(f"  Aggregated RMSE           : {rmse_loss:.4f} MPa")
            print(f"{'='*50}\n")

        return mse_loss, rmse_loss, all_predictions, all_targets

    def save(self, filename: str) -> None:
        """Saves the best model state and optimizer states to a file."""
        torch.save({
            'epoch': self._epoch,
            'best_val': self._best_val,
            'model_state_dict': self._model.state_dict(),
            'best_state_dict': self._best_state_dict,
            'optimizer_state_dict': self._optimizer.state_dict()
        }, filename)

    def load(self, filename: str) -> None:
        """Loads a saved model state."""
        checkpoint = torch.load(filename, weights_only=False)
        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._best_state_dict = checkpoint['best_state_dict']
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Important: Move optimizer state to the correct device
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)


def train_and_save_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                         val_loader: torch.utils.data.DataLoader, 
                         weight_path: str, csv_path: str, 
                         lr: float = 1e-3, epochs: int = 100000, 
                         patience: int = 1000) -> Tuple[Trainer, float, int]:
    print(f"Starting training... Destination for weights: {weight_path}")
    print(f"Hyperparameters: LR={lr}, Epochs={epochs}, Patience={patience}")
    
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer=optimizer)

    start_time = time.time()
    trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
    total_time_sec = time.time() - start_time

    epochs_run = len(trainer.train_losses)
    time_per_epoch_sec = total_time_sec / epochs_run if epochs_run > 0 else 0.0 

    trainer.save(weight_path)

    history_data = np.column_stack((trainer.train_losses, trainer.val_losses))
    header = (
        f"Hyperparameters - LR: {lr} | Patience: {patience} | Max Epochs: {epochs}\n"
        f"Total_Time_sec: {total_time_sec:.4f}\n"
        f"Epochs_run: {epochs_run}\n"
        f"Time_per_epoch_sec: {time_per_epoch_sec:.6f}\n"
        f"Train_Loss,Val_Loss"
    )
    np.savetxt(csv_path, history_data, delimiter=",", header=header, comments="# ")

    print(f"[OK] Training completed in {total_time_sec:.2f} s ({epochs_run} epochs).")
    print(f"[OK] Data saved to: {csv_path}\n")
    
    return trainer, total_time_sec, epochs_run


def plot_convergence_from_csv(csv_path: str, ignore_initial: int = 100) -> None:
    """
    Reads convergence history from a CSV file and plots the full 
    history alongside a zoomed-in version omitting the initial noisy epochs.
    """
    print(f"Reading convergence history from: {csv_path}")
    
    data = np.loadtxt(csv_path, delimiter=',', comments='#')
    
    train_losses = data[:, 0]
    val_losses = data[:, 1]
    total_epochs = len(train_losses)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full History
    ax1.plot(train_losses, label='Training', color='#1f77b4', linewidth=1.5)
    ax1.plot(val_losses, label='Validation', color='#ff7f0e', linewidth=1.5)
    
    ax1.set_yscale('log')
    ax1.set_title('Convergence (Full History)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'MSE [MPa$^2$]')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Plot 2: Zoomed (Initial Cutoff)
    if total_epochs > ignore_initial:
        zoom_epochs = range(ignore_initial, total_epochs)
        
        ax2.plot(zoom_epochs, train_losses[ignore_initial:], label='Training', color='#1f77b4', linewidth=1.5)
        ax2.plot(zoom_epochs, val_losses[ignore_initial:], label='Validation', color='#ff7f0e', linewidth=1.5)
        
        ax2.set_yscale('log')
        ax2.set_title(f'Convergence (From Epoch {ignore_initial})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(r'MSE [MPa$^2$]')
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'Training too short for zoom', ha='center', va='center')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_stress_strain_comparison(strain_seq: np.ndarray, stress_true: np.ndarray, 
                                  stress_pred: np.ndarray, sample_idx: int = 0, 
                                  component: int = 0) -> None:
    """
    Plota a curva de tensão x deformação comparando o RVE original com a PRNN.
    component: 0 -> 11, 1 -> 22, 2 -> 33, 3 -> 12, 4 -> 23, 5 -> 13
    """
    comp_names = [r"\sigma_{11}", r"\sigma_{22}", r"\sigma_{33}", r"\tau_{12}", r"\tau_{23}", r"\tau_{13}"]
    strain_names = [r"\varepsilon_{11}", r"\varepsilon_{22}", r"\varepsilon_{33}", r"\gamma_{12}", r"\gamma_{23}", r"\gamma_{13}"]

    eps = strain_seq[sample_idx, :, component]
    sig_true = stress_true[sample_idx, :, component]
    sig_pred = stress_pred[sample_idx, :, component]

    plt.figure(figsize=(7, 5))
    plt.plot(eps, sig_true, 'k-', linewidth=2.0, label='Ground Truth (RVE/Abaqus)')
    plt.plot(eps, sig_pred, 'r--', linewidth=2.0, label='PRNN Prediction')

    plt.xlabel(f"${strain_names[component]}$")
    plt.ylabel(f"${comp_names[component]}$ [MPa]")
    plt.title(f"Response Comparison - Trajectory {sample_idx}")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



def inspect_human_readable_weights(cell: Any, num_points: int) -> None:
    """
    Imprime as matrizes de pesos do Encoder (A_ij) e Decoder (W_ij)
    usando indexação matricial formal 6x6 por ponto material.
    """
    col_indices = [f"Col {j}" for j in range(1, 7)]

    print("\n" + "=" * 78)
    print("MATRIZES DE PESOS TREINADOS DA PRNN (ÁLGEBRA LINEAR CONSTITUTIVA)")
    print("=" * 78)

    # 1. ENCODER: Matriz A^(k) [6 x 6]
    enc_weight = getattr(cell.encoder, "weight", getattr(getattr(cell.encoder, "linear", None), "weight", None))
    if enc_weight is None:
        print("[Aviso] Camada linear do encoder não encontrada.")
        return
        
    enc_np = enc_weight.detach().cpu().numpy()  # Formato: [Np * 6, 6]

    print("\n[ENCODER] Matriz A^(k): eps_micro^(k) = A^(k) @ eps_macro")
    for k in range(num_points):
        print(f"\n>> Ponto Material k = {k + 1} | Matriz A^({k+1}) [6x6]:")
        block = enc_np[k * 6 : (k + 1) * 6, :]
        
        # Cabeçalho: Colunas j = 1..6
        header = "          " + " ".join([f"{c:>9}" for c in col_indices])
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(block):
            row_str = " ".join([f"{val:9.4f}" for val in row])
            # Linha i = 1..6 -> A_i1, A_i2, ...
            print(f" Linha {i+1} | {row_str}")

    # 2. DECODER: Matriz W^(k) [6 x 6]
    dec_weight = getattr(cell.decoder, "weight", getattr(getattr(cell.decoder, "linear", None), "weight", None))
    if dec_weight is None:
        print("[Aviso] Camada linear do decoder não encontrada.")
        return

    # Ativação física Softplus para obter os pesos efetivos de homogeneização
    eff_dec = F.softplus(dec_weight).detach().cpu().numpy()  # Formato: [6, Np * 6]

    print("\n" + "-" * 78)
    print("[DECODER] Matriz W^(k) = Softplus(pesos): sig_macro = sum_k W^(k) @ sig_micro^(k)")
    for k in range(num_points):
        print(f"\n>> Ponto Material k = {k + 1} | Matriz W^({k+1}) [6x6]:")
        block = eff_dec[:, k * 6 : (k + 1) * 6]
        
        header = "          " + " ".join([f"{c:>9}" for c in col_indices])
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(block):
            row_str = " ".join([f"{val:9.4f}" for val in row])
            # Linha i = 1..6 -> W_i1, W_i2, ...
            print(f" Linha {i+1} | {row_str}")

    print("=" * 78 + "\n")
