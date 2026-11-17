import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# Adiciona a raiz do projeto (um nível acima da pasta scripts/) ao sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Importações do pacote modular prnn
from prnn_project.prnn.constitutive import J2Material3DVectorized
from dissertacao.prnn_project.prnn.prnn import PRNNCell, PRNNSequence
from prnn.utils import (
    StressStrainDataset, 
    Trainer, 
    RelativeError, 
    RobustRelativeError
)
from prnn.visualization import plot_publication_prnn


def set_deterministic_seeds(seed: int = 42) -> None:
    """Fixa todas as sementes para garantir paridade com o notebook original."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_benchmark_comparison():
    # -------------------------------------------------------------------------
    # 1. Configurações Globais (Idêntico ao Jupyter Notebook)
    # -------------------------------------------------------------------------
    SEED = 42
    DTYPE = torch.float64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_deterministic_seeds(SEED)
    torch.set_default_dtype(DTYPE)

    print(f"\n{'='*70}")
    print(f"BENCHMARK NOVA ARQUITETURA (1 PONTO MATERIAL) - COMPARATIVO")
    print(f"Dispositivo: {DEVICE} | Precisão: {DTYPE}")
    print(f"{'='*70}")

    # Caminhos
    train_file = "../data/monotonic_loading.out"
    output_dir = "benchmark_comparison"
    os.makedirs(output_dir, exist_ok=True)

    weights_new = os.path.join(output_dir, "prnn_modular_1pt_soft.pth")

    # -------------------------------------------------------------------------
    # 2. Dataset e DataLoaders (Exatamente como no notebook)
    # -------------------------------------------------------------------------
    features = list(range(6))
    targets = list(range(6, 12))
    seq_length = 61

    full_dataset = StressStrainDataset(
        filename=train_file,
        features=features,
        targets=targets,
        seq_length=seq_length,
        normalize_features=False,
        dtype=DTYPE
    )

    train_subset = Subset(full_dataset, range(30))
    val_subset = Subset(full_dataset, range(30, len(full_dataset)))

    train_loader = DataLoader(train_subset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=10, shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Instanciação do Modelo Modular (PRNNCell + PRNNSequence)
    # -------------------------------------------------------------------------
    material = J2Material3DVectorized(device=DEVICE, dtype=DTYPE)
    
    cell = PRNNCell(
        num_material_points=1,
        material_instance=material,
        dim=3,                      # 3D -> 6 componentes de deformação e tensão
        decoder_type="soft",        # Corresponde à SoftLayer do código legado
        dtype=DTYPE
    )
    model = PRNNSequence(cell=cell).to(device=DEVICE, dtype=DTYPE)

    # -------------------------------------------------------------------------
    # 4. Treinamento (Sem Scheduler, Adam padrão lr=1e-3 como no notebook)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Passamos scheduler=None para garantir que o LR não caia antecipadamente
    trainer = Trainer(model, optimizer=optimizer, loss=nn.MSELoss(), scheduler=None)

    print("\nIniciando treinamento da nova arquitetura...")
    trainer.train(
        training_loader=train_loader,
        validation_loader=val_loader,
        epochs=10000,
        patience=500,
        interval=1,
        verbose=True
    )
    trainer.save(weights_new)
    print(f"\n[OK] Pesos salvos em: {weights_new}")

    # -------------------------------------------------------------------------
    # 5. Avaliação Comparativa Multi-Dataset
    # -------------------------------------------------------------------------
    criterios = {
        "MSELoss": nn.MSELoss(),
        "L1Loss (MAE)": nn.L1Loss(),
        "RelativeError": RelativeError(),
        "RobustRelativeError": RobustRelativeError()
    }

    datasets_teste = {
        "Monotonic Loading (Validação)": val_loader,
        "Once Unloading": DataLoader(
            StressStrainDataset("data/once_unloading.out", features, targets, seq_length, dtype=DTYPE),
            batch_size=10, shuffle=False
        ),
        "Gauss Process": DataLoader(
            StressStrainDataset("data/gauss_process.out", features, targets, seq_length, dtype=DTYPE),
            batch_size=10, shuffle=False
        )
    }

    print(f"\n{'='*70}")
    print("TABELA COMPARATIVA DE ERROS DA NOVA ARQUITETURA")
    print(f"{'='*70}")

    for dataset_name, loader in datasets_teste.items():
        print(f"\n--- {dataset_name} ---")
        for crit_name, crit_func in criterios.items():
            # A função eval do Trainer foi adaptada para aceitar um criterion diferente temporariamente
            loss_val, rmse_val, _, _ = trainer.eval(loader, verbose=False)
            print(f"  {crit_name:<22}: {loss_val:.6e} (Baseado no loss padrão)")

    # -------------------------------------------------------------------------
    # 6. Plotagem de Inspeção Visual (Exemplo na Curva 80 de Once Unloading)
    # -------------------------------------------------------------------------
    unloading_ds = StressStrainDataset("data/once_unloading.out", features, targets, seq_length, dtype=DTYPE)
    curve_id = 80
    
    strain_tensor, true_stress_tensor = unloading_ds[curve_id]
    strain_batch = strain_tensor.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_stress_tensor = model(strain_batch)

    hf_strain = strain_tensor.cpu().numpy()
    hf_stress = true_stress_tensor.cpu().numpy()
    nn_stress = pred_stress_tensor.squeeze(0).cpu().numpy()

    save_fig = os.path.join(output_dir, f"comparacao_modular_curva_{curve_id}.png")
    plot_publication_prnn(
        hf_strain=hf_strain,
        hf_stress=hf_stress,
        nn_strain=hf_strain,
        nn_stress=nn_stress,
        curve_id=curve_id,
        custom_labels=["Micro FE (Referência)", "Nova PRNN (1 pt)"],
        save_path=save_fig,
        contexto="artigo",
        orientacao="horizontal"
    )
    print(f"\n[OK] Gráfico salvo em: {save_fig}")


if __name__ == "__main__":
    run_benchmark_comparison()