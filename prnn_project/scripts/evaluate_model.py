"""
evaluate.py

Dedicated evaluation and post-processing script for trained PRNN checkpoints.
Uses `prnn.checkpoint.load_model` to automatically reconstruct model topology
(sparse, point, consistent/tc, dense) from saved metadata, computes physical error
metrics, and exports publication-ready 6-component Voigt figures.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensures `prnn` package is importable regardless of working directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prnn import (
    J2Material3DVectorized,
    RelativeError,
    RobustRelativeError,
    StressStrainDataset,
    load_model,
)
from prnn.material import MaterialSpec
from prnn.visualize import plot_publication_prnn


class Evaluator:
    """
    Evaluates trained PRNN models without gradient tracking, computes physical
    accuracy metrics (MSE, RMSE, Relative Error), and generates publication figures.
    """

    def __init__(
        self,
        weight_path: str,
        materials: Optional[MaterialSpec] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        num_threads: int = 4,
    ):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Checkpoint file not found: {weight_path}")

        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.weight_path = weight_path

        if self.device.type == "cpu":
            torch.set_num_threads(num_threads)
        torch.set_default_dtype(self.dtype)

        # Default J2 material if none is provided explicitly
        mat_spec = materials if materials is not None else J2Material3DVectorized(
            device=self.device, dtype=self.dtype
        )

        # Automatically reconstructs architecture & loads best weights via checkpoint.py
        self.model = load_model(self.weight_path, material_factory=mat_spec)
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()

        self.num_material_points = self.model.cell.num_material_points
        self.encoder_type = self.model.cell.encoder_type
        self.decoder_type = self.model.cell.decoder_type

        print(
            f"[OK] Loaded model from '{self.weight_path}' | "
            f"Pts: {self.num_material_points} | Enc: {self.encoder_type} | Dec: {self.decoder_type}\n"
        )

    def evaluate_metrics(
        self,
        data_loaders: Dict[str, DataLoader],
        criterion: Optional[nn.Module] = None,
        verbose_batches: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluates test/validation DataLoaders and reports per-batch and aggregated losses.
        """
        eval_criterion = criterion if criterion is not None else nn.MSELoss()
        results: Dict[str, float] = {}

        print("\n" + "=" * 70)
        print("INÍCIO DA AVALIAÇÃO DE TESTE / VALIDAÇÃO")
        print("=" * 70)

        with torch.no_grad():
            for dataset_name, loader in data_loaders.items():
                print(f"\n--- Avaliando: {dataset_name} ({len(loader)} lotes) ---")

                combined_loss = 0.0
                total_batches = len(loader)

                for j, (x, t) in enumerate(loader):
                    x = x.to(self.device, non_blocking=True)
                    t = t.to(self.device, non_blocking=True)

                    y = self.model(x)
                    loss_val = eval_criterion(y, t).item()
                    combined_loss += loss_val

                    if verbose_batches:
                        print(f"Loss for test batch {j + 1} / {total_batches} : {loss_val:.6e}")

                avg_loss = combined_loss / total_batches
                results[dataset_name] = avg_loss
                print(f"Aggregated test set loss ({dataset_name}): {avg_loss:.6e}\n")

        print("=" * 70)
        return results

    def plot_curves(
        self,
        dataset: StressStrainDataset,
        curve_ids: List[int],
        output_dir: str = "plots_eval",
        dataset_label: str = "Micro FE",
        show: bool = False,
    ) -> None:
        """
        Extracts selected load trajectories and exports 6-component Voigt panels.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGerando gráficos para as curvas {curve_ids}...")

        with torch.no_grad():
            for c_id in curve_ids:
                if c_id >= len(dataset):
                    print(f"Aviso: Curva {c_id} está fora dos limites do dataset.")
                    continue

                strain_tensor, true_stress_tensor = dataset[c_id]
                strain_batch = strain_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

                pred_stress_tensor = self.model(strain_batch)

                hf_strain = strain_tensor.cpu().numpy()
                hf_stress = true_stress_tensor.cpu().numpy()
                nn_stress = pred_stress_tensor.squeeze(0).cpu().numpy()

                save_path = os.path.join(output_dir, f"curva_{c_id}_painel_6comp.png")
                plot_publication_prnn(
                    hf_strain=hf_strain,
                    hf_stress=hf_stress,
                    nn_strain=hf_strain,
                    nn_stress=nn_stress,
                    curve_id=c_id,
                    custom_labels=[
                        dataset_label,
                        f"PRNN ({self.num_material_points} pts - {self.decoder_type})",
                    ],
                    save_path=save_path,
                    contexto="artigo",
                    orientacao="horizontal",
                    show=show,
                )
                print(f" -> Salvo: {save_path}")


def main() -> None:
    weight_file = "benchmark_points/GP-prnn_pts5_consistent_seed42.pth"
    device = torch.device("cpu")
    dtype = torch.float64

    # 1. Instancia o Avaliador (lê arquitetura automaticamente do .pth)
    evaluator = Evaluator(
        weight_path=weight_file,
        device=device,
        dtype=dtype,
        num_threads=4,
    )

    # 2. Carrega o Dataset (ordem Voigt: [xx, yy, zz, xy, yz, xz])
    features = list(range(6))
    targets = list(range(6, 12))
    seq_len = 61

    monotonic_ds = StressStrainDataset(
        "data/monotonic_loading.out", features, targets, seq_len, dtype=dtype
    )
    val_monotonic = Subset(monotonic_ds, range(30, len(monotonic_ds)))

    test_loaders = {
        "Monotonic Loading (MSE)": DataLoader(val_monotonic, batch_size=10, shuffle=False),
    }

    # 3. Métricas (MSE e Erro Relativo Robusto)
    evaluator.evaluate_metrics(test_loaders, criterion=nn.MSELoss(), verbose_batches=True)
    evaluator.evaluate_metrics(
        {"Monotonic Loading (Relative Error)": DataLoader(val_monotonic, batch_size=10, shuffle=False)},
        criterion=RobustRelativeError(sigma_ref=1.0),
        verbose_batches=False,
    )

    # 4. Geração dos Gráficos de Publicação
    evaluator.plot_curves(
        dataset=monotonic_ds,
        curve_ids=[30, 50],
        output_dir="plots_eval/monotonic",
        dataset_label="Micro FE (Monotonic)",
        show=False,
    )


if __name__ == "__main__":
    main()